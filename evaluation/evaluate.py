from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, x_test, y_test):

  y_pred = model.predict(x_test)
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)
  class_report = classification_report(y_test, y_pred)

  print("Model Evaluation Results : ")
  print(f"Accuracy : {accuracy*100:.2f} %")
  print("Confusion Matrix : \n", conf_matrix)
  print("Classification Report : \n", class_report)

  return { "accuracy" : accuracy, "confusion_matrix" : conf_matrix, "classification report" : class_report }

def log_evaluation_results(results, log_file='evaluation_log.txt'):

  with open(log_file, 'a') as log:
    log.write(f"Model Evaluation Results : \n")
    log.write(f"Accuracy : {results['accuracy']*100:.2f} %\n")
    log.write("Confusion Matrix : \n")
    log.write(f"{results['confusion_matrix']}\n")
    log.write("Classification Report : \n")
    log.write(f"{results['classification report']}\n")
    log.write("-"*40+"\n")
