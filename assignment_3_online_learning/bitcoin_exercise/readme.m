%% ABOUT THE DATA

load coin_data;

% loads the coin data.
% matrices are of size n * d, n = 213 days, d = 5 coins
%
% contains:
% s0: initial worth of the coins at day t = 0
% s:  level of coins on day t at closing time, in USD, for days t = 1,...,T
% r:  growth of the coin on day t, compared to the previous day (t-1), computed at closing times
% d:  the date of that day (t), t=1 means 2017-07-24
% 
% symbols_str: the symbol of the coin
% symbols:
% BCH = bitcoin cash
% BTC = bitcoin
% ETH = ethereum
% LTC = litecoin
% XRP = ripple

%% VISUALIZE THE DATA
% you can visualize the coindata as follows

figure
plot(s)
legend(symbols_str)
title('worth of coins')
xlabel('date')
ylabel('USD')

%% WHAT DO YOU NEED TO IMPLEMENT
%
% for exercise 4c you need to implement:
% - AA.m
%
% for exercise 4d you need to implement:
% - OGD.m
% - mix_loss.m
%
% for your convenience we have provided you:
% - loss_fixed_action.m
% - project_to_simplex.m
% these can be used in exercise 4d.
%
% the dataset is given in
% - coin_data.mat