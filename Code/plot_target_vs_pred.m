targets = targets';
predictions = permute(predictions,[2,1,3]);
predictions = reshape(predictions,[],11);
targets = reshape(targets,[],1);

figure;
for i =1:11
    plot(predictions(1:end,i));hold on;
end
plot(targets,'*-',"color",'b');hold off;
legend("p0","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","targets");
% title('2 ReLU 2 linear lyrs weight 8 testing on 2023 May');


figure;
for i =1:10
    plot(predictions(:,i));hold on;
end
plot(predictions(:,11));hold off;
% plot(targets','*-',"color",'b');hold off;
legend("p0","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11")

p = reshape(predictions',[],11);