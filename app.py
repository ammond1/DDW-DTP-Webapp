from flask import Flask, render_template, request
import numpy as np 
from ml import beta_p, predict_linreg, r2_p, mse_p, rmse_p, mins_p, maxs_p, a_p_r2, apply_power
app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')
@app.route("/our model", methods = ["GET","POST"])
def our_model():
    pred = 0
    if request.method == 'POST':
        form_data = [float(value) for key, value in request.form.items()]

        form_data = np.array(form_data).reshape(1,-1)
        print(form_data)
        form_data = apply_power(form_data)
        print(form_data)
        
        # form_data = [[3.144391e+16, 3.268362e+08, 41423.348951, 3.916254e+10,  5.549311e+11, 1.209030e+20, 21.744216,  8.557489e+12, 4415.484872, 1.244178e+10, 3.402902e+16, 250875.798159
        #               ,260.735152  ]]
        pred = float(predict_linreg(form_data, beta_p, mins_p, maxs_p))
        
    return render_template('our_model.html', r2= r2_p, mse = mse_p, rmse= rmse_p, pred = pred, a_p_r2=a_p_r2)

if __name__ == "__main__":
    app.run()