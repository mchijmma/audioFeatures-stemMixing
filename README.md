# audioFeatures-stemMixing

Processing individual stems from raw recordings is one of the first steps of multitrack audio mixing. In this work,
we explore which set of low-level audio features are sufficient to design a prediction model for this transformation.
We extract a large set of audio features from bass, guitar, vocal and keys raw recordings and stems. We show that a
procedure based on random forests classifiers can lead us to reduce significantly the number of features and we use
the selected audio features to train various multi-output regression models. Thus, we investigate stem processing
as a content-based transformation, where the inherent content of raw recordings leads us to predict the change of
feature values that occurred within the transformation.

[Paper 143rd AES](https://marquetem.files.wordpress.com/2017/09/aes-143-martinez-ramirez.pdf)
