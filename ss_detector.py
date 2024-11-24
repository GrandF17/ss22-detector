# BigData libs:
import pandas as pd

# ML:
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# AI Model:
# from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# GUI:
import tkinter as tk
from tkinter import filedialog, messagebox

# Other libs:
import numpy as np
import threading
import socket

class ShadowsocksDetector:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1000x700")
        self.root.title("Shadowsocks2022 Detector")
        self.selected_model = tk.StringVar(value="Random Forest")
        self.result_label = tk.Label(root, text="Result: ")

        self.model = None
        self.columns = None
        self.socket_thread = None
        self.socket_running = False

        # UI Elements
        self.load_button = tk.Button(root, text="Load CSV", command=self.load_csv)

        # switcher for model
        tk.Label(root, text="Select model:", font=("Arial", 14)).pack(pady=10)
        tk.Radiobutton(root, text="Random Forest", variable=self.selected_model, value="Random Forest", font=("Arial", 12)).pack(anchor="w", padx=20)
        tk.Radiobutton(root, text="Decision Tree", variable=self.selected_model, value="Decision Tree", font=("Arial", 12)).pack(anchor="w", padx=20)

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.start_socket_button = tk.Button(root, text="Start Live Connection", command=self.start_socket)
        self.stop_socket_button = tk.Button(root, text="Stop Live Connection", command=self.stop_socket, state=tk.DISABLED)

        # Layout
        self.load_button.pack(pady=5)
        self.train_button.pack(pady=5)
        self.start_socket_button.pack(pady=5)
        self.stop_socket_button.pack(pady=5)
        self.result_label.pack(pady=5)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        
        try:
            self.data = pd.read_csv(file_path)
            if 'is_ss22' not in self.data.columns:
                messagebox.showerror("Error", "CSV must contain a 'is_ss22' column.")
                return
            self.columns = [col for col in self.data.columns if col != 'is_ss22']
            messagebox.showinfo("Success", f"Data loaded with {len(self.data)} samples.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def train_model(self):
        if not hasattr(self, 'data'):
            messagebox.showerror("Error", "Load CSV first.")
            return
        
        X = self.data[self.columns]
        y = self.data['is_ss22']

        # ===================================

        if self.selected_model.get() == "Random Forest":
            self.model = RandomForestClassifier()
        elif self.selected_model.get() == "Decision Tree":
            self.model = DecisionTreeClassifier()
        else:
            messagebox.showinfo("Error", "Select model!")
            return
        
        # educating...
        test_size=0.6
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.model.fit(X_train, y_train)
        # ===================================

        # Predictions and metrics
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)

        # Classification report and confusion matrix
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)

        # drawing UI
        text_widget = tk.Text(self.root, wrap="word", width=50, height=15, padx=10, pady=10)
        text_widget.pack(padx=20, pady=20)

        text_widget.tag_config("header", font=("Arial", 12, "bold"), foreground="blue")
        text_widget.tag_config("metric", font=("Arial", 10), foreground="black")
        text_widget.tag_config("confusion", font=("Courier", 10), foreground="green")

        text_widget.insert("1.0", f"{self.selected_model.get()} performance: \n", "header")
        text_widget.insert("end", f"Accuracy: {accuracy:.4f}\n", "metric")
        text_widget.insert("end", f"Precision: {report['1']['precision']:.4f}\n", "metric")
        text_widget.insert("end", f"Recall: {report['1']['recall']:.4f}\n", "metric")
        text_widget.insert("end", f"F1-Score: {report['1']['f1-score']:.4f}\n", "metric")
        text_widget.insert("end", f"Cross-Validation Mean Accuracy: {np.mean(cv_scores):.4f}\n", "metric")
        text_widget.insert("end", "Confusion Matrix:\n", "header")
        text_widget.insert("end", f"{np.array(confusion)}", "confusion")

        # read only
        text_widget.config(state="disabled")

        # save error analysis and the percentage influence 
        # of each metric on the model chosce
        self.save_errors(X_test, y_test, y_pred)
        self.save_decision_metrics()

    
    def save_decision_metrics(self):
        if not self.model or not self.columns:
            print("Model is not trained or columns are not defined.")
            return

        importances = self.model.feature_importances_
        metrics = pd.DataFrame({
            'Feature': self.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        metrics.to_json("metrics.json", index=True)

    def save_errors(self, X_test, y_test, y_pred):
        import pandas as pd
        errors = pd.DataFrame(X_test)
        errors['Actual'] = y_test
        errors['Predicted'] = y_pred
        errors = errors[errors['Actual'] != errors['Predicted']]

        # saving false 'positive and' 'false negative' model
        # decisions made on test samples as csv file in root
        errors.to_csv("error_analysis.csv", index=False)

    def start_socket(self):
        if not self.model or not self.columns:
            messagebox.showerror("Error", "Train the model first.")
            return

        self.socket_running = True
        self.socket_thread = threading.Thread(target=self.listen_socket)
        self.socket_thread.start()
        self.start_socket_button.config(state=tk.DISABLED)
        self.stop_socket_button.config(state=tk.NORMAL)
        messagebox.showinfo("Info", "Live connection started.")

    def stop_socket(self):
        self.socket_running = False
        self.socket_thread.join()
        self.start_socket_button.config(state=tk.NORMAL)
        self.stop_socket_button.config(state=tk.DISABLED)
        messagebox.showinfo("Info", "Live connection stopped.")

    # def listen_socket(self):
    #     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     server_socket.bind(('0.0.0.0', 9999))
    #     server_socket.listen(1)
    #     while self.socket_running:
    #         client_socket, addr = server_socket.accept()
    #         data = client_socket.recv(1024).decode('utf-8')
    #         client_socket.close()

    #         # Process received data
    #         if data:
    #             self.process_live_data(data)
    #     server_socket.close()
    def listen_socket(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', 9999))
        server_socket.listen(5)
        print("Listening for connections...")

        try:
            while self.socket_running:
                client_socket, addr = server_socket.accept()
                print(f"New connection from {addr}")

                # Обрабатываем клиента в отдельном потоке
                threading.Thread(
                    target=self.handle_client,
                    args=(client_socket,),
                    daemon=True
                ).start()
        except Exception as e:
            print(f"Socket error: {e}")
        finally:
            server_socket.close()

    def handle_client(self, client_socket):
        try:
            while self.socket_running:
                data = client_socket.recv(1024).decode('utf-8')                   
                if not data:
                    print("closed")
                    break
                
                self.process_live_data(data)
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()

    def process_live_data(self, data):
        try:
            # Parse the data
            input_data = np.array([float(x) for x in data.split(",")]).reshape(1, -1)

            # Check if the number of metrics matches
            if input_data.shape[1] != len(self.columns):
                self.result_label.config(text="Error: Mismatched metrics!")
                return
            
            # print("kek")
            
            # Predict using the model
            prediction = self.model.predict(input_data)[0]
            probability = self.model.predict_proba(input_data)[0]
            result_text = f"Prediction: {prediction}, Probabilities: {probability}"
            print(result_text)
            # self.result_label.config(text=result_text)
        except Exception as e:
            self.result_label.config(text=f"Error: {e}")

def main():
    root = tk.Tk()
    app = ShadowsocksDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
