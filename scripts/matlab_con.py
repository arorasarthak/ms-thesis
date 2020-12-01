import matlab.engine
import numpy as np
import zmq


print("Starting ZMQ stuff...")
port = "6667"
recv_context = zmq.Context()
receiver = recv_context.socket(zmq.PULL)
receiver.connect("tcp://localhost:%s" % port)

snd_context = zmq.Context()
sender = snd_context.socket(zmq.PUSH)
sender.bind("tcp://*:%s" % "6668")



print("Starting Matlab Engine...")

mtlb = matlab.engine.start_matlab()
state = mtlb.eval('[0;0;0;0]')
stateCov = mtlb.eval('0.1*eye(4)')
constvel = mtlb.eval('@constvel')
cvmeas = mtlb.eval('@cvmeas')
constveljac = mtlb.eval('@constveljac')
cvmeasjac = mtlb.eval('@cvmeasjac')
state_covariance = mtlb.eval("'StateCovariance'")
num_particles = mtlb.eval("'NumParticles'")
has_additiv_noise = mtlb.eval("'HasAdditiveProcessNoise'")
true = mtlb.eval('true')
pf = mtlb.trackingPF(constvel, cvmeas, state, state_covariance, stateCov, num_particles, 1000, has_additiv_noise, true)
ukf = mtlb.trackingUKF(constvel, cvmeas, state,'Alpha', 1e-2)
EKF = mtlb.trackingEKF(constvel,cvmeas, state, 'StateTransitionJacobianFcn', constveljac, 'MeasurementJacobianFcn', cvmeasjac)


vx = mtlb.eval('0.0')
vy = mtlb.eval('0.0')
T = mtlb.eval('0.05')


try:
    while True:
        msg = receiver.recv_pyobj()
        
        if np.sum(msg) != 0.0:
            prediction = mtlb.predict(EKF, T)

            z = np.zeros((3, ))
            z = "[" + str(msg.flatten()[0]) + "," + str(msg.flatten()[1]) + "," + "0" + "]"
            z = mtlb.eval(z)
            #print(z)
            correction = mtlb.correct(EKF, z)    #.replace(",",";")
        else:
            prediction = mtlb.predict(EKF, T)

        sender.send_pyobj(np.array(prediction))

except KeyboardInterrupt:
    print("Closiing")

#
# vx = 0.0;
# vy = 0.0;
# T  = 0.3;


# pos = [5:vy*T:6;5:vy*T:6;zeros(1,4)]';
# %load('test_var_hard.mat');
# %pos_new = horzcat(test_var(:,1),test_var(:,2),zeros(518,1));
# pos_new = horzcat(filt_nn_x(284:499),filt_nn_y(284:499),zeros(216,1));
# data = zeros(215,4);

# for k = 1:size(pos_new,1)
    
#     if ~isnan(pos_new(k,:))
#         [xPred,pPred] = predict(pf,T);
#         [xCorr,pCorr] = correct(pf,(pos_new(k,:)));
#     else
#         [xPred,pPred] = predict(pf,T);       
#     end
#     data(k,:) = xCorr;
    
# end
#