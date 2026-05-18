# ═══════════════════════════════════════════════════════════════
#  KNN-2  ·  Flask Backend
#  Features : 20 numerical + gender (male/female) + age_group (young/old)
#  Auto-trains from gender.csv if no saved model is found
# ═══════════════════════════════════════════════════════════════

from flask import Flask, request, jsonify, render_template
import pickle, os, numpy as np, pandas as pd
from sklearn.neighbors       import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, r2_score

app = Flask(__name__)

# ── File paths ────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(BASE, 'gender.csv')
MODEL_PATH  = os.path.join(BASE, 'knn_model.pkl')
SCALER_PATH = os.path.join(BASE, 'scaler.pkl')
META_PATH   = os.path.join(BASE, 'encoders.pkl')

# ── Globals ───────────────────────────────────────────────────
model         = None
scaler        = None
encoders      = {}
MODEL_SCORE   = None
BEST_K        = None
IS_CLASSIFIER = None
TARGET_COL    = None
FEATURE_COLS  = None


# ═══════════════════════════════════════════════════════════════
#  TRAIN FROM CSV
# ═══════════════════════════════════════════════════════════════
def train_from_csv():
    global model, scaler, encoders, MODEL_SCORE, BEST_K
    global IS_CLASSIFIER, TARGET_COL, FEATURE_COLS

    print("\n📂 Loading dataset:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print(f"   Shape   : {df.shape}")
    print(f"   Columns : {list(df.columns)}")

    # Last column = target  (change if different in your CSV)
    TARGET_COL   = df.columns[-1]
    FEATURE_COLS = list(df.columns[:-1])
    print(f"   Target  : {TARGET_COL}")

    # Encode any string columns that are FEATURES
    encoders = {}
    cat_feat_cols = [c for c in FEATURE_COLS if df[c].dtype == object]
    for col in cat_feat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"   Encoded feature '{col}': {list(le.classes_)}")

    # Encode target if it is a string
    if df[TARGET_COL].dtype == object:
        IS_CLASSIFIER = True
        le_t = LabelEncoder()
        df[TARGET_COL] = le_t.fit_transform(df[TARGET_COL].astype(str))
        encoders['__target__'] = le_t
        print(f"   Target is categorical → KNeighborsClassifier")
    else:
        n_unique = df[TARGET_COL].nunique()
        IS_CLASSIFIER = n_unique <= 20
        kind = "KNeighborsClassifier" if IS_CLASSIFIER else "KNeighborsRegressor"
        print(f"   Target has {n_unique} unique values → {kind}")

    df = df.dropna()
    X = df[FEATURE_COLS].astype(float).values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Find best K
    print("\n🔍 Searching for best K (1–15) …")
    best_score, best_k = -9999, 5
    for k in range(1, 16):
        if IS_CLASSIFIER:
            m = KNeighborsClassifier(n_neighbors=k)
            m.fit(X_tr, y_train)
            s = accuracy_score(y_test, m.predict(X_te))
        else:
            m = KNeighborsRegressor(n_neighbors=k)
            m.fit(X_tr, y_train)
            s = r2_score(y_test, m.predict(X_te))
        print(f"   K={k:2d}  score={s:.4f}")
        if s > best_score:
            best_score, best_k = s, k

    BEST_K      = best_k
    MODEL_SCORE = best_score
    print(f"\n✅ Best K = {BEST_K}  |  Score = {MODEL_SCORE:.4f}")

    # Train final model
    if IS_CLASSIFIER:
        model = KNeighborsClassifier(n_neighbors=BEST_K)
    else:
        model = KNeighborsRegressor(n_neighbors=BEST_K)
    model.fit(X_tr, y_train)

    # Save
    with open(MODEL_PATH,  'wb') as f: pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    with open(META_PATH,   'wb') as f:
        pickle.dump({
            'encoders':      encoders,
            'is_classifier': IS_CLASSIFIER,
            'target_col':    TARGET_COL,
            'feature_cols':  FEATURE_COLS,
            'best_k':        BEST_K,
            'score':         MODEL_SCORE,
        }, f)
    print("💾 Saved: knn_model.pkl, scaler.pkl, encoders.pkl\n")


# ═══════════════════════════════════════════════════════════════
#  LOAD FROM DISK
# ═══════════════════════════════════════════════════════════════
def load_from_disk():
    global model, scaler, encoders, MODEL_SCORE, BEST_K
    global IS_CLASSIFIER, TARGET_COL, FEATURE_COLS

    with open(MODEL_PATH,  'rb') as f: model  = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(META_PATH,   'rb') as f: meta   = pickle.load(f)

    encoders      = meta['encoders']
    IS_CLASSIFIER = meta['is_classifier']
    TARGET_COL    = meta['target_col']
    FEATURE_COLS  = meta['feature_cols']
    BEST_K        = meta['best_k']
    MODEL_SCORE   = meta['score']
    print(f"✅ Loaded saved model  |  K={BEST_K}  |  Score={MODEL_SCORE:.4f}")


# ═══════════════════════════════════════════════════════════════
#  STARTUP — train or load
# ═══════════════════════════════════════════════════════════════
if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, META_PATH]):
    load_from_disk()
else:
    print("⚠️  No saved model — training from gender.csv …")
    train_from_csv()


# ═══════════════════════════════════════════════════════════════
#  HELPER — build feature vector from request body
# ═══════════════════════════════════════════════════════════════
def build_vector(data):
    """
    Supports two JSON formats:

    Format A (from the HTML page):
      {
        "features":  [24, 172.6, 82.5, ... 20 values],
        "gender":    "male",
        "age_group": "young"
      }

    Format B (raw column dict):
      { "Feature_01": 24, "gender": "male", "age_group": "young", ... }
    """
    row = {}

    if 'features' in data:
        # Format A — positional numerical list + named categoricals
        num_vals  = list(data['features'])
        gender    = str(data.get('gender', 'male')).lower()
        age_group = str(data.get('age_group', 'young')).lower()

        # Numerical columns (those without an encoder)
        num_cols = [c for c in FEATURE_COLS if c not in encoders]
        for i, col in enumerate(num_cols):
            row[col] = float(num_vals[i]) if i < len(num_vals) else 0.0

        # Categorical columns
        for col, raw_val in [('gender', gender), ('age_group', age_group)]:
            if col in encoders:
                try:
                    row[col] = float(encoders[col].transform([raw_val])[0])
                except ValueError:
                    # unknown label → use 0
                    row[col] = 0.0
            elif col in FEATURE_COLS:
                row[col] = 0.0

    else:
        # Format B — raw dict
        for col in FEATURE_COLS:
            val = data.get(col, 0.0)
            if col in encoders:
                try:
                    val = float(encoders[col].transform([str(val)])[0])
                except ValueError:
                    val = 0.0
            row[col] = float(val)

    vec = [row.get(c, 0.0) for c in FEATURE_COLS]
    return np.array(vec, dtype=float).reshape(1, -1)


# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/accuracy')
def accuracy():
    """Called by the HTML page on load to show accuracy + K value."""
    return jsonify({
        'accuracy':   round(float(MODEL_SCORE), 4),
        'best_k':     int(BEST_K),
        'model_type': 'classifier' if IS_CLASSIFIER else 'regressor',
        'target':     TARGET_COL,
        'n_features': len(FEATURE_COLS),
    })


# Both /predict and /api/predict work (HTML uses /api/predict)
@app.route('/predict',     methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST body example:
    {
      "features": [24,172.6,82.5,27.7,46.4,100.9,89.5,1.128,
                   101.0,16.0,25.0,39.2,15.6,121.6,77.0,40.7,
                   67.0,115.0,11.2,1.244],
      "gender":    "male",
      "age_group": "young"
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON received'}), 400

        X_scaled = scaler.transform(build_vector(data))
        raw_pred = model.predict(X_scaled)[0]

        if '__target__' in encoders:
            prediction = encoders['__target__'].inverse_transform([int(raw_pred)])[0]
        elif IS_CLASSIFIER:
            prediction = int(raw_pred)
        else:
            prediction = round(float(raw_pred), 4)

        return jsonify({'prediction': prediction, 'k': BEST_K})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/retrain', methods=['POST'])
def retrain():
    """Force retrain from CSV (POST to /retrain)."""
    try:
        train_from_csv()
        return jsonify({'message': 'Retrained successfully',
                        'best_k': BEST_K, 'score': MODEL_SCORE})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/info')
def info():
    """Debug: see all model metadata."""
    return jsonify({
        'best_k':        BEST_K,
        'score':         MODEL_SCORE,
        'is_classifier': IS_CLASSIFIER,
        'target_col':    TARGET_COL,
        'feature_cols':  FEATURE_COLS,
        'encoders':      {k: list(v.classes_)
                          for k, v in encoders.items() if k != '__target__'},
    })


# ═══════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"\n🌐  Visit: http://127.0.0.1:5000/")
    print(f"📊  Model  : {'Classifier' if IS_CLASSIFIER else 'Regressor'}")
    print(f"🎯  Score  : {MODEL_SCORE:.4f}")
    print(f"🔢  Best K : {BEST_K}")
    print(f"📋  Features: {len(FEATURE_COLS)}\n")
    app.run(debug=True)