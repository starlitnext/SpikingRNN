function [left_data_set, right_data_set] = load_data()
load('data/DATA.mat')
left=find(Y_DATA_POSTPRE==0);
right=find(Y_DATA_POSTPRE==1);

left_data_set = cell(1, length(left));
right_data_set = cell(1, length(right));

 for i=1:length(left)
        left_data_set{i} = (X_DATA_BASECOR{1,left(i)});
end
for i=1:length(right)
        right_data_set{i} = (X_DATA_BASECOR{1,right(i)});
end

end