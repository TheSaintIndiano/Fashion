# Sportswear Classification

Sportswear or activewear is clothing, including footwear, worn for sport or physical exercise. Sport-specific clothing is worn for most sports and physical exercise, for practical, comfort or safety reasons.
Typical sport-specific garments include shorts, tracksuits, T-shirts, tennis shirts and polo shirts.
Specialized garments include swimsuits (for swimming), wetsuits (for diving or surfing), ski suits (for skiing) and leotards (for gymnastics).

Sports footwear include trainers, football boots, riding boots, and ice skates. Sportswear also includes some underwear, such as the jockstrap and sports bra.

Sportswear is also at times worn as casual fashion clothing.

# Models

Sportswear classification has been divided into different modules based of Sklearn, keras machine learning frameworks. I also have used Auto-Sklearn, TPOT auto-ml for exhaustive optimizations and parameter space searches.

Algorithms used are as follow.
Sklearn : Xtreme Gradient Boosting, K Nearest Neighbours, Support Vector Machines, Naive Bayes

Keras : LSTMs with Embedding and/or TF-IDF 

# Run Scripts


### Auto-Sklearn

###### Folder : models/auto-sklearn/

Training:

`python run.py --root_dir "./" --data_dir "../../data/sportswear/events" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/auto-sklearn" --save_dir "../../save/auto-sklearn" --test_size 33`

Sampling:

`python run.py --root_dir "./" --data_dir "../../data/sportswear/events" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/auto-sklearn" --save_dir "../../save/auto-sklearn" --test_size 33 --run_type sample`

### TPOT

###### Folder : models/auto-tpot/

Training:

`python run.py --root_dir "./" --data_dir "../../data/sportswear/events" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/auto-tpot" --save_dir "../../save/auto-tpot" --test_size 33`

Sampling:

`python sample.py --root_dir "./" --data_dir "../../data/sportswear/events" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/auto-tpot" --save_dir "../../save/auto-tpot" --test_size 33`

### Classical models (XGB, KNN, Naive Bayes, Support Vector)

###### Folder : models/classical/

`python run.py --root_dir "./" --raw_data_dir "../../data/raw/sportswear/events" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/classical" --save_dir "../../save/classical" --model xgb --test_size 33`


### Keras

###### Folder : models/keras/

1. Using Keras Tokenizer with Embedding layer
2. Using Keras Tokenizer with TF-IDF 
3. Using Keras with TF-IDF only

Training:

`python run.py --root_dir "./" --data_dir "../../data/keras" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/keras" --save_dir "../../save/keras" --test_size 33 `

Sampling:

`python run.py --root_dir "./" --data_dir "../../data/keras" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/keras" --save_dir "../../save/keras" --test_size 33 --run_type sample`

### Results

The accuracy of 99% has been acheived using XGB, Keras LSTMs with Embedding. The best peforming model is XGB due to less computation usage and time complexity.

