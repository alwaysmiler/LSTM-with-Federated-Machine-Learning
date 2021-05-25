# LSTM-with-Federated-Machine-Learning
1. Install 64 bit Python V3.8
2. Install tensorflow and other required package
3. Make sure the data path in TFClass1.py and TFClass2.py, TFClass3.py is path to dataset 1, 2, 3
4. Run Server.py-->it will output the weights update for client1 and client2, client3 for each iteration
5. Run Client1.py, Client2.py, Client3.py -> it will output the loss, RMSE, NRMSE for each client.
6. Currently the stop criteria has not be been set up. It will continue to run. Final accuracy will be approaching 1 eventually and weights converge (weights do not change at local client and the weights of two client are close to each other).
