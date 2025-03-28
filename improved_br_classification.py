import tkinter as tk
from tkinter import ttk
import threading
import time
import pandas as pd
import numpy as np
import re
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from scipy.sparse import hstack, csr_matrix # For merging


def run_evaluation(project, result_label, progressbar):
    def process():
        start_time = time.time()

        result_label.config(text="Running evaluation... Please wait.")
        progressbar.start() # Animation

        try:
            df = pd.read_csv(f"{project}.csv")
            df.fillna("", inplace=True)

            # Remove backets/quotes
            def remove_junk(txt): return re.sub(r'[\[\]\"\']', ' ', txt.strip())

            # Merge to one
            def merge_CSV_fields(row):
                parts = [row['Title'], row['Body'], row['Labels'], row['Comments'], row['Codes'], row['Commands']]
                return " ".join([remove_junk(str(p)) for p in parts])

            df['merged_text'] = df.apply(merge_CSV_fields, axis=1)
            y = df['class'].values # Target labels

            def clean_text(text):
                text = text.lower()
                text = re.sub(r"http\S+|www.\S+", " ", text)
                text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
                tokens = word_tokenize(text)
                lemmatizer = WordNetLemmatizer()
                sw = set(stopwords.words('english'))
                # Useful error terms
                important_words = {"not", "should", "might", "could", "would", "can't", "won't", "error", "issue"}
                for word in important_words:
                    sw.discard(word) # Discarded stopwords

                #sw.discard('not')
                return [lemmatizer.lemmatize(t) for t in tokens if t not in sw and len(t) > 1]

            df['tokens'] = df['merged_text'].apply(clean_text)

            tfidf = TfidfVectorizer(
                tokenizer=lambda x: x, # tokenized already
                lowercase=False,
                ngram_range=(1, 2),
                max_features=3000
            )
            X_tfidf = tfidf.fit_transform(df['tokens'])

            glove_dim = 100
            glove_path = "./glove.6B.100d.txt"
            glove_dict = {}
            with open(glove_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    glove_dict[parts[0]] = np.array(parts[1:], dtype='float32')

            def avg_glove_vector(tokens):
                vectors = [glove_dict[t] for t in tokens if t in glove_dict]
                return np.mean(vectors, axis=0) if vectors else np.zeros(glove_dim)

            X_glove = np.vstack(df['tokens'].apply(avg_glove_vector))
            X_glove_sparse = csr_matrix(X_glove)

            def count_words(txt): return len(txt.strip().split())
            def count_labels(txt): return len(remove_junk(txt).split())

            meta = pd.DataFrame({
                'state_is_closed': df['State'].apply(lambda s: 1 if s.lower() == 'closed' else 0),
                'length_of_body': df['Body'].apply(count_words),
                'num_labels': df['Labels'].apply(count_labels),
                'num_comments': df['Comments'].apply(lambda x: len(remove_junk(x).split())),
                'length_of_title': df['Title'].apply(count_words)
            })

            meta_scaled = StandardScaler().fit_transform(meta)

            X_meta_sparse = csr_matrix(meta_scaled)
            X_full = hstack([X_tfidf, X_glove_sparse, X_meta_sparse])

            REPEAT = 30
            test_ratio = 0.3
            accs, precs, recs, f1s, aucs = [], [], [], [], []

            for seed in range(REPEAT):
                train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_ratio, random_state=seed)
                X_train, X_test = X_full[train_idx], X_full[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                lr_params = {
                    'C': [1],  # keep fast, or expand to [0.01, 0.1, 1, 10] for full
                    'penalty': ['l2'],
                    'solver': ['lbfgs'],
                    'class_weight': ['balanced']
                }
                rf_params = {
                    'n_estimators': [100],
                    'max_depth': [None],
                    'class_weight': ['balanced']
                }

                # Grid search logistic regression. increase cv for better results but slow
                grid_lr = GridSearchCV(LogisticRegression(max_iter=3000), lr_params, cv=3, scoring='f1', n_jobs=-1)
                grid_lr.fit(X_train, y_train)

                # Grid search random forest. increase cv for better results but slow
                grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='f1', n_jobs=-1)
                grid_rf.fit(X_train, y_train)

                # LR / RF best one chosen
                best_model = grid_lr.best_estimator_ if grid_lr.best_score_ >= grid_rf.best_score_ else grid_rf.best_estimator_
                y_pred = best_model.predict(X_test)

                accs.append(accuracy_score(y_test, y_pred))
                precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
                recs.append(recall_score(y_test, y_pred, average='macro'))
                f1s.append(f1_score(y_test, y_pred, average='macro'))

                # Get predicted probabilities for ROC/AUC calculation
                y_proba = best_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
                auc_val = auc(fpr, tpr)
                aucs.append(auc_val)

                # fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
                # aucs.append(auc(fpr, tpr))

            elapsed_time = time.time() - start_time

            result_text = (
                f"\nResults for: {project}\n"
                f"Repeats: {REPEAT} | Test Size: {int(test_ratio*100)}%\n"
                f"Accuracy:  {np.mean(accs):.4f}\n"
                f"Precision: {np.mean(precs):.4f}\n"
                f"Recall:    {np.mean(recs):.4f}\n"
                f"F1 Score:  {np.mean(f1s):.4f}\n"
                f"AUC:       {np.mean(aucs):.4f}\n\n"
                f"Time Taken: {elapsed_time:.2f} seconds"
            )

            result_label.config(text=result_text)

            # Save results to CSV
            out_csv_name = f"./results_{project}_HYBRID.csv"  # Save per project

            try:
                pd.read_csv(out_csv_name, nrows=1)
                header_needed = False
            except:
                header_needed = True

            df_log = pd.DataFrame({
                'Time_Seconds': [elapsed_time],
                'repeated_times': [REPEAT],
                'Accuracy': [np.mean(accs)],
                'Precision': [np.mean(precs)],
                'Recall': [np.mean(recs)],
                'F1': [np.mean(f1s)],
                'AUC': [np.mean(aucs)],
                'CV_list(AUC)': [str(aucs)],
                'CV_list(F1)': [str(f1s)]
            })

            df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)
            result_text += f"\n\nResults saved to: {out_csv_name}"
            result_label.config(text=result_text)

        except Exception as e:
            result_label.config(text="Error: " + str(e))
        finally:
            progressbar.stop()

    threading.Thread(target=process).start() # Thread so GUI stays responsive


# GUI
root = tk.Tk()
root.title("Bug Report Classifier (Hybrid TF-IDF + GloVe + Meta + LR / RF)")
root.geometry("1000x600")
root.resizable(False, False)

tk.Label(root, text="=== Bug Report Classifier - HYBRID: TF-IDF + GloVe + Meta-Features -> LR / RF (Best F1) ===", font=('Arial', 13)).pack(pady=10)
tk.Label(root, text="Select Dataset:", font=('Arial', 12)).pack(pady=10)

dataset_options = ["pytorch", "tensorflow", "keras", "incubator-mxnet", "caffe"]
dataset_var = tk.StringVar(value=dataset_options[0])
dropdown = ttk.Combobox(root, textvariable=dataset_var, values=dataset_options, state="readonly", font=('Arial', 12))
dropdown.pack()

run_button = tk.Button(root, text="Run Evaluation", font=('Arial', 12), command=lambda: run_evaluation(dataset_var.get(), result_label, progress))
run_button.pack(pady=15)

progress = ttk.Progressbar(root, mode='indeterminate', length=300) # Useless progress bar
progress.pack()

result_label = tk.Label(root, text="Results will appear here.", justify="center", font=('Arial', 11), anchor="center")
result_label.pack(padx=10, pady=(10,0), fill='both', expand=True)

close_button = tk.Button(root, text="Close", font=('Arial', 12), command=root.destroy)
close_button.pack(pady=10)


root.mainloop()
