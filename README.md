# Sportswear Classification

Sportswear or activewear is clothing, including footwear, worn for sport or physical exercise. Sport-specific clothing is worn for most sports and physical exercise, for practical, comfort or safety reasons.
Typical sport-specific garments include shorts, tracksuits, T-shirts, tennis shirts and polo shirts.
Specialized garments include swimsuits (for swimming), wetsuits (for diving or surfing), ski suits (for skiing) and leotards (for gymnastics).

Sports footwear include trainers, football boots, riding boots, and ice skates. Sportswear also includes some underwear, such as the jockstrap and sports bra.

Sportswear is also at times worn as casual fashion clothing.

# Models

Sportswear classification has been divided into different modules based of Sklearn, keras machine learning frameworks. I also have used Auto-Sklearn, TPOT auto-ml for exhaustive optimizations and parameter space searches.

Algorithms used in different modules are as follow.

* Sklearn : Xtreme Gradient Boosting, K Nearest Neighbours, Support Vector Machines, Naive Bayes
* Keras   : LSTMs with Embedding and/or TF-IDF 
* Auto-Sklearn : Bayesian optimization, meta-learning and ensemble construction.
* TPOT    :  Genetic programming.

# Run Scripts

>Every model has training & sampling scripts. After preprocessing the raw data, an hdf file containing URL texts and their labels will be created during the initialization phase of training.
Folder indicates the location of each model's scripts.

### Auto-Sklearn


###### Folder : models.auto-sklearn/

_Training_:

`python run.py --data_dir "../../data/sportswear/events" --checkpoint_dir "../../checkpoint/auto-sklearn" --save_dir "../../save/auto-sklearn"`

_Sampling_:

`python run.py --hdf_file "../../data/hdf/sportswear" --checkpoint_dir "../../checkpoint/auto-sklearn" --save_dir "../../save/auto-sklearn" --run_type sample`

### TPOT

###### Folder : models/auto-tpot/

_Training_:

`python run.py --data_dir "../../data/sportswear/events" --checkpoint_dir "../../checkpoint/auto-tpot" --save_dir "../../save/auto-tpot"`

_Sampling_:

`python sample.py --hdf_file "../../data/hdf/sportswear" --checkpoint_dir "../../checkpoint/auto-tpot" --save_dir "../../save/auto-tpot"`

### Classical models (XGB, KNN, Naive Bayes, Support Vector)

###### Folder : models/classical/

`python run.py --data_dir "../../data/sportswear/events" --checkpoint_dir "../../checkpoint/classical" --save_dir "../../save/classical"`


### Keras

###### Folder : models/keras/

1. Using Keras Tokenizer with Embedding layer
2. Using Keras Tokenizer with TF-IDF 
3. Using Keras with TF-IDF only

_Training_:

`python run.py --data_dir "../../data/sportswear/events" --checkpoint_dir "../../checkpoint/keras" --save_dir "../../save/keras"`

_Sampling_:

`python run.py --hdf_file "../../data/hdf/sportswear" --checkpoint_dir "../../checkpoint/keras" --save_dir "../../save/keras" --run_type sample`

### Results

The following table shows a model accuracy and time taken respectively.

| Model                   | Accuracy     | Time        |
|:-----------------------:|:------------:|:-----------:|
| `XGB`                   | `96.28`      | Few Minutes |
| Naive Bayes             | 90.01        | Few Minutes |
| KNN                     | 90.35        | Few Minutes |
| `Support Vector Machine`| `98.78`      | Few Minutes |
| Auto-Sklearn            | 99.17        | 2 hours |
| TPOT                    | 98.95        | 5 hours |
| **Keras with Embedding**| **99.51**    | **5 mintutes** |



The accuracy of 99% has been acheived using Support Vector Machine, Keras LSTMs with Embedding. The best peforming model is Support Vector Machine due to less computation usage and time complexity.


### Further Improvements

* Balance dataset using different resampling techniques.
* Exploring different kinds of embedding (and training with our data to customize to our requirements) for more semantic and contextual understanding of URL description feature as I have used tf-idf vectorization for topic/document modelling.
* Analysing special keywords for our task and giving them more weights during training/classification.
* Use Image as another input feature along with a semantic understanding of Color, Size features & Amount as a good discriminating numerical feature.
* The above rich features will help improve the model and we can further train Ensemble and/or Deep learning models to increase accuracy with the availability of a large amount of data over the time.
