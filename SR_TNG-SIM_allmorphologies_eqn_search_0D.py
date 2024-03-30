import time
import numpy as np
import pandas as pd

import galsim #install with conda install -c conda_forge galsim

import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.cm as cm
import matplotlib.colors as norm
from matplotlib.gridspec import SubplotSpec
import seaborn as sns

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline #This allows one to build different steps together
from sklearn.preprocessing import StandardScaler, RobustScaler

# from tqdm import tqdm 

from pysr import PySRRegressor

import os
# os.mkdir('TNG-SIM_images/v27_SR_eqn_search_allmorph/SR_v1_allmorph0D_eqn_search')

#### Physical Model Equation search in TNG-SIM dataset, all morphologies galaxies 

# Normalized dataset: all masses divided by halo mass (Mvir)
df_0D = pd.read_csv('/Users/festabu/Desktop/ML_galaxy_size_project/Codes/TNG-SIM_images/v3_initial_analysis/v6_TNG-SIM_0D_Dataset_wMvir.csv')

df_sample = df_0D.sample(n = 8000, random_state = 2023) 

# choose a subset of randomly sampled data, 
# fix random seed here because we don't want to have the random seed effect when we repeat the eqn search with more iterations
# choose a subset of 8,000 randomly sampled galaxies because Symbolic Regression takes a max of 10K entries


# choose only the 5 most important features from SVR feature ranking (since SVR predictions are better than RF),
# in order to reduce the time to run SR modelling

#  1 SubhaloVmaxRad_hs_log 0.3078147600888657
#  2 SubhaloMH2_hs_log 0.4955123319168074
#  3 SubhaloStarMetallicity_log 0.5847325527497041
#  4 SubhaloMstar_hs_log 0.7115393108197582
#  5 Group_M_TopHat200_log 0.7933021654446595

X_imp = df_sample.loc[:, [ 'SubhaloVmaxRad_hs', 'SubhaloMH2_hs', 'SubhaloStarMetallicity', 
                          'SubhaloMstar_hs', 
                          'Group_M_TopHat200'      
                                      ]]


y_imp = df_sample.loc[:, 'SubhaloRstar_hs']

# choose the Symbolic Regression model; choose the mathematical operations allowed
model_imp = PySRRegressor(
    
    niterations=10000,
    
    unary_operators=[ "square"], #, "exp", "cube", "log10_abs", "log1p"] ,#   "inv(x) = 1/x"
#         "inv(x) = 1/x",  # Custom operator (julia syntax)
         
    
    binary_operators=["+", "-", "*", "pow", "/"], #"mylogfunc(x)=log(1-(x))" ],


    constraints={
        "pow": (4, 1),
        "/": (-1, 4),
#         "log1p": 4,
    },
    
    # extra_sympy_mappings={'mylogfunc': lambda x: log1p(x)},
    
    nested_constraints={
        "pow": {"pow": 0}, #, "exp": 0},
        "square": {"square": 0} #, "cube": 0, "exp": 0},
#         "cube": {"square": 0, "cube": 0, "exp": 0},
#         "exp": {"square": 0, "cube": 0, "exp": 0},
#         "log1p": {"pow": 0, "exp": 0},
    },
    
    maxsize=30,
    multithreading=False,
    model_selection="best", # Result is mix of simplicity+accuracys
#     loss="loss(x, y) = (x - y)^2"  # Custom loss function (julia syntax)

#     procs=7
)


start_time = time.time()

model_imp.fit(X_imp, np.array(y_imp))

elapsed_time = time.time() - start_time
elapsed_time_min = elapsed_time/60
elapsed_time_hr = elapsed_time/3600

print(f"Elapsed time to compute the SymbolicRegression fitting for TNG-SIM, all morphologies, 0D+Mvir dataset: {elapsed_time:.3f} seconds, {elapsed_time_min:.3f} minutes, or {elapsed_time_hr:.3f} hours")

# Run3 with n_iter=10,000, n_galaxies = 8,000, random_state fixed, unary and binary operators included 
# (unary_operators=[ "square"],  binary_operators=["+", "-", "*", "pow", "/"], constraints={"pow": (4, 1), "/":(-1, 4)}
#  nested_constraints={ "pow": {"pow": 0}, "square": {"square": 0}   ,
# loss function is custom, as in one of Miles' examples, loss="loss(x, y) = (x - y)^2"



model_imp.equations_
allmorphologies_eqns = model_imp.equations_
allmorphologies_eqns.to_csv('TNG-SIM_images/v27_SR_eqn_search_allmorph/SR_v1_allmorph0D_eqn_search/run3_SR_allmorphologies_equations_n_iter_10K')

allmorphologies_pred = model_imp.predict(X_imp)
allmorphologies_pred = pd.DataFrame(allmorphologies_pred)
allmorphologies_pred.to_csv('TNG-SIM_images/v27_SR_eqn_search_allmorph/SR_v1_allmorph0D_eqn_search/run3_SR_allmorphologies_Predicted_sizes_n_iter_10K')


print(model_imp.sympy())


r2_score_allmorph=r2_score(y_imp, model_imp.predict(X_imp))


with open('TNG-SIM_images/v27_SR_eqn_search_allmorph/SR_v1_allmorph0D_eqn_search/run3_SR_allmorphologies_bestequation_n_iter_10K.txt', 'w', encoding='utf-8') as txt_save:
    txt_save.write('The most important features used by SR to find the best equation for All Morphologies galaxies are extracted from the first 5 best features from SVR feature ranking run on 0D+Mvir dataset: \n \n')
    txt_save.write(str(X_imp.columns.to_list()))
    txt_save.write('\n \n')
    txt_save.write('Run3: The best equation with n_iter 10,000 is:')
    txt_save.write(str(model_imp.sympy()))
    txt_save.write('\n \n')
    txt_save.write(r'$R^{2}$ score=' + '{:.2f}'.format(r2_score_allmorph))
    txt_save.write('\n \n')
    txt_save.write('Elapsed time to compute the All Morphologies Symbolic Regression =' + '{:.3f}'.format(elapsed_time) + 'seconds \n' + 'That is {:.3f}'.format(elapsed_time_min) + 'minutes \n' + '{:.3f}'.format(elapsed_time_hr) + 'hours \n' )
    txt_save.write('Run3 with n_iter=10,000, n_galaxies = 8,000, random_state=2023 fixed, unary and binary operators included (unary_operators=[ "square"],  binary_operators=["+", "-", "*", "pow", "/"], constraints={"pow": (4, 1), "/":(-1, 4)}; nested_constraints={ "pow": {"pow": 0}, "square": {"square": 0}) and default loss function')




plt.scatter(y_imp, model_imp.predict(X_imp),
            c = df_sample.SubhaloVmax_hs,  cmap='Spectral_r',
            s=10, marker='.', alpha=0.7, label= r'$\frac{V_{max}}{V_{vir}}$', vmin=1.15, vmax=1.65) #,label= label,
plt.axis([0.0,0.06, 0.0,0.06])
plt.plot([-3.0, 1.0], [-3.0, 1.0], color = 'black', linewidth = 2)
plt.text(0.0015, 0.05, r'$R^{2}$ score=' + '{:.2f}'.format(r2_score_allmorph), size=12)
plt.text(0.0015, 0.03, 'eqn=' + '{}'.format(model_imp.sympy()), size=10)
plt.title('Predicted vs True 0D+Mvir Galaxy Size with SR \n' + r'All morphologies')
plt.xlabel('True Rstar/Rvir')
plt.ylabel('Predicted Rstar/Rvir ')
plt.legend(loc='lower right' , shadow=True)
plt.colorbar()
plt.savefig('TNG-SIM_images/v27_SR_eqn_search_allmorph/SR_v1_allmorph0D_eqn_search/run3_SR_allmorphologies0D_predicted_vs_true_size_n_iter_10K.jpeg', dpi=500)
plt.show()


