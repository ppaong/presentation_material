# training loss

### pretraining loss graph
![1][pre_loss]

data : 7(mwu119126, mwu120212, mwu120414, mwu121921, mwu125222, mwu126325, mwu126426)    
batch size : 2   
epoch : 30   


### fine tunning loss graph
![2][fine_loss]

data : 3(mwu102311,mwu127226,mwu128026)    
batch size : 1    
epoch : 30   



# applying

### fine tunning data
* mwu102311    
* mean squared error: 0.0015456372042010676    
* mwu127226    
* mean squared error: 0.0015794248234955064    
* mwu128026    
* mean squared error: 0.0010406949393452524    

### validation data
* mwu165032
* mean squared error: 0.002560754671256886
* mwu168240
* mean squared error: 0.0027351746728604175



# sampling

### mwu102311(fine tunning data)      
| |x|y|z|
|---|---|---|---|
|predict|![x][ex_x_p]|![y][ex_y_p]|![z][ex_z_p]|
|target|![x][ex_x_t]|![y][ex_y_t]|![z][ex_z_t]|



### mwu165032(validation data)   
| |x|y|z|
|---|---|---|---|
|predict|![x][val_x_p]|![y][val_y_p]|![z][val_z_p]|
|target|![x][val_x_t]|![y][val_y_t]|![z][val_z_t]|





[pre_loss]: ../slide8_DiffustionMRI/pretraining.png
[fine_loss]: ../slide8_DiffustionMRI/fine_tuning.png

[ex_x_p]: ../slide8_DiffustionMRI/example_data/mwu102311_predict_sagittal_x_axis_animation.gif
[ex_y_p]: ../slide8_DiffustionMRI/example_data/mwu102311_predict_coronal_y_axis_animation.gif
[ex_z_p]: ../slide8_DiffustionMRI/example_data/mwu102311_predict_axial_z_axis_animation.gif

[ex_x_t]: ../slide8_DiffustionMRI/example_data/mwu102311_target_sagittal_x_axis_animation.gif
[ex_y_t]: ../slide8_DiffustionMRI/example_data/mwu102311_target_coronal_y_axis_animation.gif
[ex_z_t]: ../slide8_DiffustionMRI/example_data/mwu102311_target_axial_z_axis_animation.gif

[val_x_p]: ../slide8_DiffustionMRI/validation_data/mwu165032_predict_sagittal_x_axis_animation.gif
[val_y_p]: ../slide8_DiffustionMRI/validation_data/mwu165032_predict_coronal_y_axis_animation.gif
[val_z_p]: ../slide8_DiffustionMRI/validation_data/mwu165032_predict_axial_z_axis_animation.gif

[val_x_t]: ../slide8_DiffustionMRI/validation_data/mwu165032_target_sagittal_x_axis_animation.gif
[val_y_t]: ../slide8_DiffustionMRI/validation_data/mwu165032_target_coronal_y_axis_animation.gif
[val_z_t]: ../slide8_DiffustionMRI/validation_data/mwu165032_target_axial_z_axis_animation.gif

