// Binary classification of the Breast Cancer Wisconsin dataset with a
// from-scratch scalar reverse-mode autograd engine.
//
// This is a single-file C++ port of the three Python files
//
//     engine.py                 -> the `Value` autograd type
//     nn.py                     -> Neuron / Layer / MLP
//     cancer_prediction.py      -> loss, training, cross-validation, main
//
// keeping everything in one translation unit. The only runtime input is the
// CSV file `breast_cancer.csv` (30 feature columns + a 0/1 `diagnosis`
// column); the per-fold training curves are written to
// `cancer_training_cpp.csv` instead of being drawn with matplotlib.
//
// Build:  g++ -O2 -std=c++17 cancer_prediction.cpp -o cancer_prediction
// Run:    ./cancer_prediction
//
// Task
// ----
// Predict whether a tumor is malignant from 30 numeric features computed from a
// digitized image of a fine needle aspirate (FNA) of a breast mass.
//
//     label 1 = malignant  (positive class, 212 samples)
//     label 0 = benign     (negative class, 357 samples)
//
// Model & loss
// ------------
// A logistic-regression / MLP classifier trained with binary cross-entropy.
// The network outputs a single logit `z`; the probability is `sigmoid(z)`.
// The loss is the numerically stable "BCE with logits":
//
//     L = softplus(z) - y * z,   where softplus(z) = log(1 + e^z)
//
// Evaluation
// ----------
// The whole dataset is evaluated with stratified k-fold cross-validation:
// every sample is used for validation exactly once, and the reported metrics
// are per-fold. Features are standardized inside each fold using only that
// fold's training statistics, so no information leaks from the validation
// samples.
//
// Note: the RNG is not bit-for-bit compatible with NumPy / Python's `random`,
// so fold assignments, weight initializations and the resulting metrics are
// statistically equivalent to, but not identical to, the Python version.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Globals (mirrors cancer_prediction.py)
// ---------------------------------------------------------------------------
constexpr int FOLDS = 5;
constexpr int EPOCHS = 60;
constexpr double LR = 0.05;
constexpr int BATCH_SIZE = 32;
constexpr double ALPHA = 1e-4;  // for L2 regularization
constexpr unsigned SEED = 1337;
const std::vector<int> HIDDEN = {16, 16, 1};

// First training curve output file (matplotlib replacement).
const std::string CURVES_CSV = "cancer_training_cpp.csv";
const std::string DATA_CSV = "breast_cancer.csv";

// RNG used for weight initialization (seeded in main).
static std::mt19937 g_rng(SEED);
static std::uniform_real_distribution<double> g_uniform(-1.0, 1.0);

// ===========================================================================
// Scalar autograd engine (port of engine.py)
// ===========================================================================
class Value;
using Val = std::shared_ptr<Value>;

// A single scalar node in the computation graph. Children are owned through
// `shared_ptr`, so the graph frees itself once the output goes out of scope.
// `local_grads[i]` is d(out)/d(prev[i]) and is computed at forward time.
struct Value : std::enable_shared_from_this<Value> {
    double data;
    double grad = 0.0;
    std::vector<Val> prev;            // children, in deterministic order
    std::string op;                   // kept for repr/debugging only
    std::vector<double> local_grads;  // per-child local derivative

    explicit Value(double d) : data(d) {}

    void backward();
    Val relu();
    Val exp();
    Val log();
    Val sigmoid();
    Val pow(double exponent);
};

inline Val make_val(double d) { return std::make_shared<Value>(d); }

// --- binary ops ------------------------------------------------------------
inline Val operator+(const Val& a, const Val& b) {
    Val out = make_val(a->data + b->data);
    out->prev = {a, b};
    out->op = "+";
    out->local_grads = {1.0, 1.0};
    return out;
}

inline Val operator*(const Val& a, const Val& b) {
    Val out = make_val(a->data * b->data);
    out->prev = {a, b};
    out->op = "*";
    out->local_grads = {b->data, a->data};
    return out;
}

inline Val operator-(const Val& a, const Val& b) {
    // self + (-other)
    return a + (b * make_val(-1.0));
}

inline Val operator/(const Val& a, const Val& b) {
    // self * other**-1
    return a * b->pow(-1.0);
}

// --- unary minus -----------------------------------------------------------
inline Val operator-(const Val& a) { return a * make_val(-1.0); }

// --- mixed Val / double overloads (Python's automatic coercion) ------------
inline Val operator+(const Val& a, double b) { return a + make_val(b); }
inline Val operator+(double a, const Val& b) { return make_val(a) + b; }
inline Val operator*(const Val& a, double b) { return a * make_val(b); }
inline Val operator*(double a, const Val& b) { return make_val(a) * b; }
inline Val operator-(const Val& a, double b) { return a - make_val(b); }
inline Val operator-(double a, const Val& b) { return make_val(a) - b; }
inline Val operator/(const Val& a, double b) { return a / make_val(b); }
inline Val operator/(double a, const Val& b) { return make_val(a) / b; }

// --- elementwise nonlinearities --------------------------------------------
inline Val Value::relu() {
    const double d = data;
    Val out = make_val(d < 0.0 ? 0.0 : d);
    out->prev = {shared_from_this()};
    out->op = "ReLU";
    out->local_grads = {d > 0.0 ? 1.0 : 0.0};
    return out;
}

inline Val Value::exp() {
    const double e = std::exp(data);
    Val out = make_val(e);
    out->prev = {shared_from_this()};
    out->op = "exp";
    out->local_grads = {e};
    return out;
}

inline Val Value::log() {
    // d/dx ln(x) = 1/x ; floor the argument so log(0) -> a finite value
    const double x = std::max(data, 1e-12);
    Val out = make_val(std::log(x));
    out->prev = {shared_from_this()};
    out->op = "log";
    out->local_grads = {1.0 / x};
    return out;
}

inline Val Value::sigmoid() {
    // 1 / (1 + e^-x), computed in a numerically stable way; grad = s * (1 - s)
    double s;
    if (data >= 0.0) {
        const double z = std::exp(-data);
        s = 1.0 / (1.0 + z);
    } else {
        const double z = std::exp(data);
        s = z / (1.0 + z);
    }
    Val out = make_val(s);
    out->prev = {shared_from_this()};
    out->op = "sigmoid";
    out->local_grads = {s * (1.0 - s)};
    return out;
}

inline Val Value::pow(double exponent) {
    Val out = make_val(std::pow(data, exponent));
    out->prev = {shared_from_this()};
    out->op = "pow";
    out->local_grads = {exponent * std::pow(data, exponent - 1.0)};
    return out;
}

// Generic reverse pass over the whole graph (children before parents).
inline void Value::backward() {
    std::vector<Value*> topo;
    std::unordered_set<Value*> visited;

    std::function<void(Value*)> build_topo = [&](Value* v) {
        if (visited.count(v) != 0) return;
        visited.insert(v);
        for (const Val& child : v->prev) build_topo(child.get());
        topo.push_back(v);
    };
    build_topo(this);

    // the entire backward pass, generic over all ops
    grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Value* v = *it;
        for (std::size_t i = 0; i < v->prev.size(); ++i) {
            v->prev[i]->grad += v->local_grads[i] * v->grad;
        }
    }
}

// Sum of a list of Values, starting from 0 (mirrors Python's `sum(...)`).
inline Val sum_values(const std::vector<Val>& values) {
    Val total = make_val(0.0);
    for (const Val& v : values) total = total + v;
    return total;
}

// ===========================================================================
// Neural network (port of nn.py)
// ===========================================================================
class Neuron {
public:
    Neuron(int nin, bool nonlin) : b_(make_val(0.0)), nonlin_(nonlin) {
        w_.reserve(nin);
        for (int i = 0; i < nin; ++i) w_.push_back(make_val(g_uniform(g_rng)));
    }

    Val operator()(const std::vector<Val>& x) const {
        Val act = b_;
        for (std::size_t i = 0; i < w_.size(); ++i) act = act + w_[i] * x[i];
        return nonlin_ ? act->relu() : act;
    }

    std::vector<Val> parameters() const {
        std::vector<Val> params = w_;
        params.push_back(b_);
        return params;
    }

    std::string repr() const {
        return std::string(nonlin_ ? "ReLU" : "Linear") +
               "Neuron(" + std::to_string(w_.size()) + ")";
    }

private:
    std::vector<Val> w_;
    Val b_;
    bool nonlin_;
};

class Layer {
public:
    Layer(int nin, int nout, bool nonlin)
        : nin_(nin), nout_(nout) {
        neurons_.reserve(nout);
        for (int i = 0; i < nout; ++i) neurons_.emplace_back(nin, nonlin);
    }

    std::vector<Val> operator()(const std::vector<Val>& x) const {
        std::vector<Val> out;
        out.reserve(neurons_.size());
        for (const Neuron& n : neurons_) out.push_back(n(x));
        return out;
    }

    std::vector<Val> parameters() const {
        std::vector<Val> params;
        for (const Neuron& n : neurons_) {
            std::vector<Val> p = n.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }

private:
    int nin_;
    int nout_;
    std::vector<Neuron> neurons_;
};

class MLP {
public:
    MLP(int nin, const std::vector<int>& nouts) {
        std::vector<int> sz = {nin};
        sz.insert(sz.end(), nouts.begin(), nouts.end());
        for (std::size_t i = 0; i < nouts.size(); ++i) {
            const bool nonlin = (i != nouts.size() - 1);
            layers_.emplace_back(sz[i], sz[i + 1], nonlin);
        }
    }

    // Returns the single output logit (the last layer has one neuron).
    Val operator()(const std::vector<Val>& x) const {
        std::vector<Val> cur = x;
        for (const Layer& layer : layers_) cur = layer(cur);
        return cur.front();
    }

    std::vector<Val> parameters() const {
        std::vector<Val> params;
        for (const Layer& layer : layers_) {
            std::vector<Val> p = layer.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }

    void zero_grad() const {
        for (const Val& p : parameters()) p->grad = 0.0;
    }

private:
    std::vector<Layer> layers_;
};

// ===========================================================================
// Loss & evaluation helpers (port of cancer_prediction.py)
// ===========================================================================
using Matrix = std::vector<std::vector<double>>;

Val bce_with_logits_loss(const MLP& model, const Matrix& X,
                         const std::vector<double>& y) {
    // Mean binary cross-entropy (with logits), as a single autograd Value.
    // We build the softplus branch explicitly so the argument of `log` is
    // always >= 1 and the result never becomes +/-inf.
    std::vector<Val> losses;
    losses.reserve(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) {
        std::vector<Val> xi;
        xi.reserve(X[i].size());
        for (double v : X[i]) xi.push_back(make_val(v));
        Val z = model(xi);

        Val softplus;
        if (z->data > 0.0) {
            // log(1 + e^z) = z + log(1 + e^-z)  (exponent stays <= 0)
            softplus = z + (make_val(1.0) + (-z)->exp())->log();
        } else {
            // log(1 + e^z)  (exponent stays <= 0)
            softplus = (make_val(1.0) + z->exp())->log();
        }

        // softplus(z) - y*z  ==  BCE(sigmoid(z), y)
        losses.push_back(softplus - make_val(y[i]) * z);
    }
    return sum_values(losses) * (1.0 / static_cast<double>(losses.size()));
}

Val l2_regularization(const MLP& model, double alpha) {
    // L2 penalty over all parameters, as a single Value.
    Val total = make_val(0.0);
    for (const Val& p : model.parameters()) total = total + p * p;
    return total * alpha;
}

double bce_from_probs(const std::vector<double>& y,
                      const std::vector<double>& probs) {
    // Mean BCE from predicted probabilities (no autograd graph).
    // Probabilities are clipped so log(0) cannot occur.
    double loss = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double p = std::min(std::max(probs[i], 1e-12), 1.0 - 1e-12);
        loss += -(y[i] * std::log(p) + (1.0 - y[i]) * std::log(1.0 - p));
    }
    return loss / static_cast<double>(y.size());
}

std::vector<double> predict_proba(const MLP& model, const Matrix& X) {
    // Return P(malignant) for each row of X.
    std::vector<double> probs;
    probs.reserve(X.size());
    for (const std::vector<double>& row : X) {
        std::vector<Val> xi;
        xi.reserve(row.size());
        for (double v : row) xi.push_back(make_val(v));
        Val z = model(xi);
        probs.push_back(z->sigmoid()->data);
    }
    return probs;
}

double accuracy(const std::vector<double>& y_true,
                const std::vector<double>& y_pred) {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

// ===========================================================================
// Training (port of cancer_prediction.py)
// ===========================================================================
struct History {
    std::vector<double> train_loss;
    std::vector<double> val_loss;
};

History train(MLP& model, const Matrix& X_tr, const std::vector<double>& y_tr,
              const Matrix& X_va, const std::vector<double>& y_va) {
    // Mini-batch SGD with a linearly decaying learning rate.
    std::mt19937 rng(SEED);
    const std::size_t n = X_tr.size();
    History history;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // learning rate decays linearly to 10% of its initial value
        const double lr = LR * (1.0 - 0.9 * epoch / std::max(1, EPOCHS - 1));

        std::vector<std::size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        double epoch_loss = 0.0;

        for (std::size_t s = 0; s < n; s += BATCH_SIZE) {
            const std::size_t e = std::min(s + BATCH_SIZE, n);

            Matrix X_batch;
            std::vector<double> y_batch;
            X_batch.reserve(e - s);
            y_batch.reserve(e - s);
            for (std::size_t k = s; k < e; ++k) {
                X_batch.push_back(X_tr[perm[k]]);
                y_batch.push_back(y_tr[perm[k]]);
            }

            Val data_loss = bce_with_logits_loss(model, X_batch, y_batch);
            Val total_loss = data_loss + l2_regularization(model, ALPHA);

            model.zero_grad();
            total_loss->backward();
            for (const Val& p : model.parameters()) p->data -= lr * p->grad;

            epoch_loss += data_loss->data * static_cast<double>(e - s);
        }

        history.train_loss.push_back(epoch_loss / static_cast<double>(n));
        history.val_loss.push_back(
            bce_from_probs(y_va, predict_proba(model, X_va)));
    }

    return history;
}

// ===========================================================================
// Data loading, stratified k-fold and standardization
// ===========================================================================
struct Dataset {
    Matrix X;
    std::vector<double> y;
};

Dataset load_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("could not open " + path);

    Dataset ds;
    std::string line;
    bool header = true;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (header) {  // first line is the column header
            header = false;
            continue;
        }
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        if (row.empty()) continue;
        ds.y.push_back(row.back());
        row.pop_back();
        ds.X.push_back(std::move(row));
    }
    return ds;
}

// Stratified k-fold: shuffle each class separately, then deal the samples into
// the folds round-robin so the class ratio stays balanced across folds.
std::vector<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>>
stratified_kfold(const std::vector<double>& y, int n_splits, unsigned seed) {
    std::mt19937 rng(seed);

    std::map<double, std::vector<std::size_t>> by_class;
    for (std::size_t i = 0; i < y.size(); ++i) by_class[y[i]].push_back(i);

    std::vector<std::vector<std::size_t>> fold_indices(n_splits);
    for (auto& [label, idx] : by_class) {
        std::shuffle(idx.begin(), idx.end(), rng);
        for (std::size_t j = 0; j < idx.size(); ++j) {
            fold_indices[j % static_cast<std::size_t>(n_splits)].push_back(idx[j]);
        }
    }

    std::vector<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>> folds;
    folds.reserve(n_splits);
    for (int f = 0; f < n_splits; ++f) {
        std::vector<std::size_t> train_idx;
        std::vector<std::size_t> val_idx;
        for (int g = 0; g < n_splits; ++g) {
            if (g == f) {
                val_idx = fold_indices[g];
            } else {
                train_idx.insert(train_idx.end(), fold_indices[g].begin(),
                                 fold_indices[g].end());
            }
        }
        folds.emplace_back(std::move(train_idx), std::move(val_idx));
    }
    return folds;
}

class StandardScaler {
public:
    void fit(const Matrix& X) {
        const std::size_t n = X.size();
        const std::size_t d = X.front().size();
        mean_.assign(d, 0.0);
        for (const auto& row : X) {
            for (std::size_t j = 0; j < d; ++j) mean_[j] += row[j];
        }
        for (double& m : mean_) m /= static_cast<double>(n);

        scale_.assign(d, 0.0);
        for (const auto& row : X) {
            for (std::size_t j = 0; j < d; ++j) {
                const double diff = row[j] - mean_[j];
                scale_[j] += diff * diff;
            }
        }
        for (double& s : scale_) {
            s = std::sqrt(s / static_cast<double>(n));
            if (s == 0.0) s = 1.0;  // constant feature: no scaling
        }
    }

    Matrix transform(const Matrix& X) const {
        Matrix out = X;
        for (auto& row : out) {
            for (std::size_t j = 0; j < row.size(); ++j) row[j] = (row[j] - mean_[j]) / scale_[j];
        }
        return out;
    }

private:
    std::vector<double> mean_;
    std::vector<double> scale_;
};

// ===========================================================================
// Main
// ===========================================================================
// Writes per-epoch mean/std of the train and validation losses across folds.
void save_curves(const std::vector<History>& histories) {
    const std::size_t n_epochs = histories.front().train_loss.size();
    const double n_folds = static_cast<double>(histories.size());

    std::ofstream out(CURVES_CSV);
    if (!out) throw std::runtime_error("could not write " + CURVES_CSV);
    out << "epoch,train_mean,train_std,val_mean,val_std\n";

    for (std::size_t e = 0; e < n_epochs; ++e) {
        double train_mean = 0.0, val_mean = 0.0;
        for (const History& h : histories) {
            train_mean += h.train_loss[e];
            val_mean += h.val_loss[e];
        }
        train_mean /= n_folds;
        val_mean /= n_folds;

        double train_var = 0.0, val_var = 0.0;
        for (const History& h : histories) {
            train_var += (h.train_loss[e] - train_mean) * (h.train_loss[e] - train_mean);
            val_var += (h.val_loss[e] - val_mean) * (h.val_loss[e] - val_mean);
        }
        train_var /= n_folds;
        val_var /= n_folds;

        out << (e + 1) << "," << train_mean << "," << std::sqrt(train_var) << ","
            << val_mean << "," << std::sqrt(val_var) << "\n";
    }
    std::cout << "saved training curve to " << CURVES_CSV << "\n";
}

int main() {
    try {
        g_rng.seed(SEED);

        // load the data
        Dataset ds = load_csv(DATA_CSV);
        const std::size_t n = ds.X.size();
        const std::size_t d = ds.X.front().size();
        std::size_t malignant = 0;
        for (double label : ds.y) {
            if (label == 1.0) ++malignant;
        }
        std::cout << "dataset: " << n << " samples, " << d << " features\n";
        std::cout << "classes: " << malignant << " malignant, " << (n - malignant)
                  << " benign\n";

        const auto folds = stratified_kfold(ds.y, FOLDS, SEED);
        std::vector<History> histories;

        for (int fold = 0; fold < FOLDS; ++fold) {
            const auto& [tr, va] = folds[fold];

            Matrix X_tr, X_va;
            std::vector<double> y_tr, y_va;
            X_tr.reserve(tr.size());
            X_va.reserve(va.size());
            y_tr.reserve(tr.size());
            y_va.reserve(va.size());
            for (std::size_t i : tr) {
                X_tr.push_back(ds.X[i]);
                y_tr.push_back(ds.y[i]);
            }
            for (std::size_t i : va) {
                X_va.push_back(ds.X[i]);
                y_va.push_back(ds.y[i]);
            }

            StandardScaler scaler;
            scaler.fit(X_tr);
            Matrix X_tr_scaled = scaler.transform(X_tr);
            Matrix X_va_scaled = scaler.transform(X_va);

            MLP model(static_cast<int>(d), HIDDEN);
            History history = train(model, X_tr_scaled, y_tr, X_va_scaled, y_va);
            histories.push_back(history);

            // calculate the accuracy of the validation set
            const std::vector<double> probs = predict_proba(model, X_va_scaled);
            std::vector<double> preds(probs.size());
            for (std::size_t i = 0; i < probs.size(); ++i) {
                preds[i] = probs[i] > 0.5 ? 1.0 : 0.0;
            }
            const double accuracy_score = accuracy(y_va, preds);

            std::cout << "fold " << (fold + 1) << "/" << FOLDS << ": "
                      << "train loss " << std::fixed << std::setprecision(4)
                      << history.train_loss.back() << ", "
                      << "val loss " << history.val_loss.back() << ", "
                      << "acc " << std::setw(5) << std::setprecision(2)
                      << accuracy_score * 100.0 << "%\n";
        }

        save_curves(histories);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
