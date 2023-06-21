Kp=40;

filename = 'step_data.csv';
M = csvread(filename,1);
time = M(:,1);
pos = M(:,3);
ref = M(:,4);

filename = 'mod_step_data.csv';
M = csvread(filename,1);
mod_time = M(:,1);
u = M(:,8);
y = M(:,3);

filename = 'mod_data_0.csv';
M = csvread(filename,1);
time0 = M(:,9);
pos0 = M(:,10);
ref0 = M(:,11);
u0 = M(:,8);

filename = 'mod_data_1.csv';
M = csvread(filename,1);
time1 = M(:,9);
pos1 = M(:,10);
ref1 = M(:,11);
u1 = M(:,8);

filename = 'mod_data_2.csv';
M = csvread(filename,1);
time2 = M(:,9);
pos2 = M(:,10);
ref2 = M(:,11);
u2 = M(:,8);

filename = 'mod_data_3.csv';
M = csvread(filename,1);
time3 = M(:,9);
pos3 = M(:,10);
ref3 = M(:,11);
u3 = M(:,8);

filename = 'mod_data_4.csv';
M = csvread(filename,1);
time4 = M(:,9);
pos4 = M(:,10);
ref4 = M(:,11);
u4 = M(:,8);

filename = 'mod_data_5.csv';
M = csvread(filename,1);
time5 = M(:,9);
pos5 = M(:,10);
ref5 = M(:,11);
u5 = M(:,8);

filename = 'mod_data_6.csv';
M = csvread(filename,1);
time6 = M(:,9);
pos6 = M(:,10);
ref6 = M(:,11);
u6 = M(:,8);

filename = 'mod_data_7.csv';
M = csvread(filename,1);
time7 = M(:,9);
pos7 = M(:,10);
ref7 = M(:,11);
u7 = M(:,8);

filename = 'mod_data_8.csv';
M = csvread(filename,1);
time8 = M(:,9);
pos8 = M(:,10);
ref8 = M(:,11);
u8 = M(:,8);

step_amplitude0=ref0(1);
step_amplitude1=ref1(1);
step_amplitude2=ref2(1);
step_amplitude3=ref3(1);
step_amplitude4=ref4(1);
step_amplitude5=ref5(1);
step_amplitude6=ref6(1);
step_amplitude7=ref7(1);
step_amplitude8=ref8(1);

K0=(step_amplitude0-(step_amplitude0-pos0(end)))/(Kp*(step_amplitude0-pos0(end)));
K1=(step_amplitude1-(step_amplitude1-pos1(end)))/(Kp*(step_amplitude1-pos1(end)));
K2=(step_amplitude2-(step_amplitude2-pos2(end)))/(Kp*(step_amplitude2-pos2(end)));
K3=(step_amplitude3-(step_amplitude3-pos3(end)))/(Kp*(step_amplitude3-pos3(end)));
K4=(step_amplitude4-(step_amplitude4-pos4(end)))/(Kp*(step_amplitude4-pos4(end)));
K5=(step_amplitude5-(step_amplitude5-pos5(end)))/(Kp*(step_amplitude5-pos5(end)));
K6=(step_amplitude6-(step_amplitude6-pos6(end)))/(Kp*(step_amplitude6-pos6(end)));
K7=(step_amplitude7-(step_amplitude7-pos7(end)))/(Kp*(step_amplitude7-pos7(end)));
K8=(step_amplitude8-(step_amplitude8-pos8(end)))/(Kp*(step_amplitude8-pos8(end)));


tau=3.3;

step_amplitude=step_amplitude6;
time=time6;
pos=pos6;
K=K6;


% run this in comand window: systemIdentification