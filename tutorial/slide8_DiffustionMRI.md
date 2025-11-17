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
mean squared error: 0.0015456372042010676     
| |x|y|z|
|---|---|---|---|
|predict|![x][ex1_x_p]|![y][ex1_y_p]|![z][ex1_z_p]|
|target|![x][ex1_x_t]|![y][ex1_y_t]|![z][ex1_z_t]|

### mwu127226(fine tunning data)    
mean squared error: 0.0015794248234955064     
| |x|y|z|
|---|---|---|---|
|predict|![x][ex2_x_p]|![y][ex2_y_p]|![z][ex2_z_p]|
|target|![x][ex2_x_t]|![y][ex2_y_t]|![z][ex2_z_t]|

---

### mwu165032(validation data)   
mean squared error: 0.002560754671256886    
| |x|y|z|
|---|---|---|---|
|predict|![x][val1_x_p]|![y][val1_y_p]|![z][val1_z_p]|
|target|![x][val1_x_t]|![y][val1_y_t]|![z][val1_z_t]|

### mwu168240(validation data)   
mean squared error: 0.0027351746728604175    
| |x|y|z|
|---|---|---|---|
|predict|![x][val2_x_p]|![y][val2_y_p]|![z][val2_z_p]|
|target|![x][val2_x_t]|![y][val2_y_t]|![z][val2_z_t]|



[pre_loss]: ../slide8_DiffustionMRI/pretraining.png
[fine_loss]: ../slide8_DiffustionMRI/fine_tuning.png

[ex1_x_p]: ../slide8_DiffustionMRI/example_data/mwu102311_predict_sagittal_x_axis_animation.gif
[ex1_y_p]: ../slide8_DiffustionMRI/example_data/mwu102311_predict_coronal_y_axis_animation.gif
[ex1_z_p]: ../slide8_DiffustionMRI/example_data/mwu102311_predict_axial_z_axis_animation.gif

[ex1_x_t]: ../slide8_DiffustionMRI/example_data/mwu102311_target_sagittal_x_axis_animation.gif
[ex1_y_t]: ../slide8_DiffustionMRI/example_data/mwu102311_target_coronal_y_axis_animation.gif
[ex1_z_t]: ../slide8_DiffustionMRI/example_data/mwu102311_target_axial_z_axis_animation.gif

[val1_x_p]: ../slide8_DiffustionMRI/validation_data/mwu165032_predict_sagittal_x_axis_animation.gif
[val1_y_p]: ../slide8_DiffustionMRI/validation_data/mwu165032_predict_coronal_y_axis_animation.gif
[val1_z_p]: ../slide8_DiffustionMRI/validation_data/mwu165032_predict_axial_z_axis_animation.gif

[val1_x_t]: ../slide8_DiffustionMRI/validation_data/mwu165032_target_sagittal_x_axis_animation.gif
[val1_y_t]: ../slide8_DiffustionMRI/validation_data/mwu165032_target_coronal_y_axis_animation.gif
[val1_z_t]: ../slide8_DiffustionMRI/validation_data/mwu165032_target_axial_z_axis_animation.gif


[ex2_x_p]: ../slide8_DiffustionMRI/example_data/mwu127226_predict_sagittal_x_axis_animation.gif
[ex2_y_p]: ../slide8_DiffustionMRI/example_data/mwu127226_predict_coronal_y_axis_animation.gif
[ex2_z_p]: ../slide8_DiffustionMRI/example_data/mwu127226_predict_axial_z_axis_animation.gif

[ex2_x_t]: ../slide8_DiffustionMRI/example_data/mwu127226_target_sagittal_x_axis_animation.gif
[ex2_y_t]: ../slide8_DiffustionMRI/example_data/mwu127226_target_coronal_y_axis_animation.gif
[ex2_z_t]: ../slide8_DiffustionMRI/example_data/mwu127226_target_axial_z_axis_animation.gif

[val2_x_p]: ../slide8_DiffustionMRI/validation_data/mwu168240_predict_sagittal_x_axis_animation.gif
[val2_y_p]: ../slide8_DiffustionMRI/validation_data/mwu168240_predict_coronal_y_axis_animation.gif
[val2_z_p]: ../slide8_DiffustionMRI/validation_data/mwu168240_predict_axial_z_axis_animation.gif

[val2_x_t]: ../slide8_DiffustionMRI/validation_data/mwu168240_target_sagittal_x_axis_animation.gif
[val2_y_t]: ../slide8_DiffustionMRI/validation_data/mwu168240_target_coronal_y_axis_animation.gif
[val2_z_t]: ../slide8_DiffustionMRI/validation_data/mwu168240_target_axial_z_axis_animation.gif



















