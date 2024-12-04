from flask import Flask, render_template, request, flash
import numpy as np 
from ml import beta_p, predict_linreg, r2_p, mse_p, rmse_p, mins_p, maxs_p, a_p_r2, apply_power
app = Flask(__name__)

# Set the secret key for session and flashing
app.secret_key = 'abcdefghijklmnop'  # Replace with a secure random string

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')
@app.route("/our model", methods = ["GET","POST"])
def our_model():
    pred = 0
    if request.method == 'POST':
        try:
            form_data = [float(value) for key, value in request.form.items()]
            form_data = np.array(form_data).reshape(1, -1)
            print(form_data)
            form_data = apply_power(form_data)
            print(form_data)
            
            pred = float(predict_linreg(form_data, beta_p, mins_p, maxs_p))
        
        except ValueError as e:
            flash("Invalid input!")
            return render_template('our_model.html', r2=r2_p, mse=mse_p, rmse=rmse_p, pred=649, a_p_r2=a_p_r2)

    return render_template('our_model.html', r2=r2_p, mse=mse_p, rmse=rmse_p, pred=pred, a_p_r2=a_p_r2)

if __name__ == "__main__":
    app.run()