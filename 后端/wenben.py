from flask import Flask,jsonify
from flask import render_template,request
import joblib
import numpy as np
from flask_cors import CORS
import torch
from torch import nn
import json

class RNN(nn.Module):
    def __init__(self,input_size):
        super(RNN,self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64,1)
        )

    def forward(self, x):
        r_out,(h_n,h_c) = self.rnn(x,None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out)
        return out



from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/wenbenwa', methods=['POST'])
def wenben():
    if request.method == "POST":
        data = request.get_json(silent=True)
        #print(data)
        work_data=data["work_data"]
        wenjuan_data=data["wenjuan_data"]
        shijian_data=data["shijian_data"]

        work_data = int(work_data)
        wenjuan_data = int(wenjuan_data)
        shijian_data = int(shijian_data)

        x = []
        x.append(work_data)
        x.append(wenjuan_data)
        x = np.array(x).reshape(1,-1)
        #print(x)
        model = joblib.load('direct_line1.pkl')
        y1 = model.predict(x)
        print(y1)
        model = joblib.load('tree1.pkl')
        y3 = model.predict(x)
        print(y3)
        model = joblib.load('integrated_learning1.pkl')
        y4 = model.predict(x)
        print(y4)
        mean=84
        std=12
        x1=(x-mean)/std
        model = joblib.load('SGD_line1.pkl')
        y2 = model.predict(x1)
        y2=y2*std+mean
        print(y2)
        m_state_dict = torch.load('model1.pt')
        model = RNN(2)
        model.load_state_dict(m_state_dict)
        mean=77
        std=12
        x2=(x-mean)/std
        x2=torch.Tensor(x2)
        y5 = model(torch.unsqueeze(x2, dim=1))  # unsqueeze(tx,dim=1)在tx的1处增加一个维度
        y5 = y5[:, -1, :]
        y5 = y5.detach().numpy()
        y5=y5*std+mean
        y5=y5.flatten()
        print(y5)
        json_data = {
            'status': True,
            'data': {
                'linear_final': str(round(float(y1), 2)),
                'SGD_final': str(round(float(y2), 2)),
                'tree_final': str(round(float(y3), 2)),
                'boost_final': str(round(float(y4), 2)),
                'deep_final':  str(round(float(y5), 2))
            }
        }
        return jsonify(json_data)

@app.route('/jisuan', methods=['POST'])
def jisuan():
    if request.method == "POST":
        data = request.get_json(silent=True)
    # print(data)
    work_data = data["work_data"]
    wenjuan_data = data["wenjuan_data"]
    shijian_data = data["shijian_data"]

    work_data = int(work_data)
    wenjuan_data = int(wenjuan_data)
    shijian_data = int(shijian_data)

    x = []
    x.append(work_data)
    x.append(wenjuan_data)
    x.append(shijian_data)
    x = np.array(x).reshape(1, -1)
    # print(x)
    model = joblib.load('direct_line2.pkl')
    y1 = model.predict(x)
    print(y1)
    model = joblib.load('tree2.pkl')
    y3 = model.predict(x)
    print(y3)
    model = joblib.load('integrated_learning2.pkl')
    y4 = model.predict(x)
    print(y4)
    mean = 72
    std = 11
    x1 = (x - mean) / std
    model = joblib.load('SGD_line2.pkl')
    y2 = model.predict(x1)
    y2 = y2 * std + mean
    print(y2)
    m_state_dict = torch.load('model2.pt')
    model = RNN(3)
    model.load_state_dict(m_state_dict)
    mean = 75
    std = 11
    x2 = (x - mean) / std
    x2 = torch.Tensor(x2)
    y5 = model(torch.unsqueeze(x2, dim=1))  # unsqueeze(tx,dim=1)在tx的1处增加一个维度
    y5 = y5[:, -1, :]
    y5 = y5.detach().numpy()
    y5 = y5 * std + mean
    y5 = y5.flatten()
    print(y5)
    json_data = {
        'status': True,
        'data': {
            'linear_final': str(round(float(y1), 2)),
            'SGD_final': str(round(float(y2), 2)),
            'tree_final': str(round(float(y3), 2)),
            'boost_final': str(round(float(y4), 2)),
            'deep_final': str(round(float(y5), 2))
        }
    }
    return jsonify(json_data)


if __name__ == '__main__':
    app.run(host="127.0.0.1",debug=True)
    #app.run(debug=True)