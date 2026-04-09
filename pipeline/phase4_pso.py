# ============================================================
# PHASE 4 - PARTICLE SWARM OPTIMIZATION
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("  PHASE 4 - PARTICLE SWARM OPTIMIZATION")
print("="*55)

# ============================================================
# LOAD DATA
# ============================================================
X_train_bal = np.load('models/X_train_bal.npy')
y_train_bal = np.load('models/y_train_bal.npy')
X_test      = np.load('models/X_test.npy')
y_test      = np.load('models/y_test.npy')

print(f"\nTraining samples : {X_train_bal.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# ============================================================
# PSO CLASS
# ============================================================
class PSO_MLP:
    def __init__(self, n_particles=10, n_iterations=15):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.w  = 0.7   # inertia
        self.c1 = 1.5   # cognitive
        self.c2 = 1.5   # social

        np.random.seed(99)
        self.positions          = np.random.uniform(0, 1, (n_particles, 4))
        self.velocities         = np.random.uniform(-0.1, 0.1, (n_particles, 4))
        self.personal_best_pos  = self.positions.copy()
        self.personal_best_score= np.zeros(n_particles)
        self.global_best_pos    = None
        self.global_best_score  = 0
        self.loss_history       = []

    def decode(self, position):
        n1 = int(8 + position[0] * 24)        # neurons layer1: 8-32
        n2 = int(4 + position[1] * 12)        # neurons layer2: 4-16
        lr = 0.0001 + position[2] * 0.0099    # learning rate: 0.0001-0.01
        al = 0.001 + position[3] * 0.099      # alpha: 0.001-0.1
        return n1, n2, lr, al

    def evaluate(self, position):
        n1, n2, lr, al = self.decode(position)
        try:
            mlp = MLPClassifier(
                hidden_layer_sizes=(n1, n2),
                learning_rate_init=lr,
                alpha=al,
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=15
            )
            scores = cross_val_score(mlp, X_train_bal, y_train_bal,
                                     cv=2, scoring='accuracy')
            return scores.mean()
        except:
            return 0.0

    def optimize(self):
        print(f"\nPSO started:")
        print(f"  Particles  : {self.n_particles}")
        print(f"  Iterations : {self.n_iterations}")
        print(f"  Inertia w  : {self.w}")
        print(f"  Cognitive  : {self.c1}")
        print(f"  Social     : {self.c2}")
        print("-" * 50)

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                score = self.evaluate(self.positions[i])
                if score > self.personal_best_score[i]:
                    self.personal_best_score[i] = score
                    self.personal_best_pos[i]   = self.positions[i].copy()
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos   = self.positions[i].copy()

            r1 = np.random.uniform(0, 1, (self.n_particles, 4))
            r2 = np.random.uniform(0, 1, (self.n_particles, 4))

            self.velocities = (
                self.w  * self.velocities +
                self.c1 * r1 * (self.personal_best_pos - self.positions) +
                self.c2 * r2 * (self.global_best_pos   - self.positions)
            )
            self.positions = np.clip(self.positions + self.velocities, 0, 1)
            self.loss_history.append(self.global_best_score)

            n1, n2, lr, _ = self.decode(self.global_best_pos)
            print(f"  Iter {iteration+1:>2}/{self.n_iterations} --- "
                  f"Best CV Acc: {self.global_best_score*100:.2f}% | "
                  f"Layers:({n1},{n2}) LR:{lr:.4f}")

        print("-" * 50)
        n1, n2, lr, al = self.decode(self.global_best_pos)
        print(f"\nPSO Complete.")
        print(f"  Best CV Accuracy : {self.global_best_score*100:.2f}%")
        print(f"  Optimal Neurons  : ({n1}, {n2})")
        print(f"  Optimal LR       : {lr:.5f}")
        print(f"  Optimal Alpha    : {al:.6f}")
        return self.global_best_pos

# ============================================================
# RUN PSO
# ============================================================
pso      = PSO_MLP(n_particles=10, n_iterations=15)
best_pos = pso.optimize()

n1, n2, lr, al = pso.decode(best_pos)

# ============================================================
# SAVE
# ============================================================
joblib.dump(pso, 'models/pso.pkl')
np.save('models/pso_best_pos.npy', best_pos)
joblib.dump({
    'n1': n1, 'n2': n2, 'lr': lr, 'al': al,
    'best_cv_accuracy': pso.global_best_score
}, 'models/pso_best_config.pkl')
print("\nPSO results saved to models/ folder.")

# ============================================================
# PLOTS
# ============================================================
# Plot 1 - PSO Convergence Curve
plt.figure(figsize=(9, 5))
plt.plot([x*100 for x in pso.loss_history], color='darkorange', linewidth=2.5)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Best CV Accuracy (%)', fontsize=12)
plt.title('PSO Convergence Curve --- Best Accuracy per Iteration', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/pso_convergence.png', dpi=150)
plt.close()
print("Saved: pso_convergence.png")

# Plot 2 - Swarm Diversity
plt.figure(figsize=(9, 5))
plt.bar(range(pso.n_particles),
        [s*100 for s in pso.personal_best_score],
        color='steelblue', edgecolor='black', alpha=0.85, label='Personal Best')
plt.axhline(y=pso.global_best_score*100, color='red', linewidth=2, linestyle='--',
            label=f'Global Best: {pso.global_best_score*100:.2f}%')
plt.xlabel('Particle Index',     fontsize=12)
plt.ylabel('Best Accuracy (%)',  fontsize=12)
plt.title('PSO --- Personal Best vs Global Best Across Particles', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('outputs/pso_swarm.png', dpi=150)
plt.close()
print("Saved: pso_swarm.png")

print("\n" + "="*55)
print("  PHASE 4 COMPLETE")
print("="*55)