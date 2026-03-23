# fbow_vocab_trainer

Standalone vocabulary trainer that:
1. Loads an image folder (optionally recursive)
2. Extracts either `orb` (binary ORB descriptors) or `sift` (float SIFT descriptors)
3. Trains a Bag-of-Words vocabulary and writes it to `output.vocab_path`

Vocabulary training backend note: this package trains vocabularies using the in-workspace `fbow` library (`fbow::VocabularyCreator`). The output file is `fbow`'s native binary format (not the `ORBvoc.txt` DBoW2 text format expected by `ORB_SLAM3` in this workspace yet).

## Config (yaml-cpp)

Run with:

`fbow_vocab_trainer --config-file <path>`

YAML keys used by the trainer:

- `feature.type`: `orb` or `sift`
- `dataset.images_dir`: root images folder
- `dataset.recursive`: optional (default `true`)
- `dataset.max_images`: optional (default `0` = no cap)
- `dataset.extensions`: optional (default `png,jpg,jpeg,bmp,tiff`)
- `orb.*`: optional ORB extractor parameters (only used when `feature.type=orb`)
- `sift.*`: optional SIFT extractor parameters (only used when `feature.type=sift`)
- `trainer.k`, `trainer.L`: fbow vocabulary tree parameters
- `trainer.nthreads`: optional (default `1`)
- `trainer.max_iters`: optional (default `0` in the sample configs)
- `trainer.max_features_per_image`: optional (default `0` = no cap)
- `output.vocab_path`: output file path
- `test.enabled`: optional (default `true`)
- `test.max_images`: optional (default `5`)
- `test.top_k`: optional (default `10`)
- `test.max_features_per_image`: optional (default `0` = use `trainer.max_features_per_image`)
- `test.output_file`: optional (default `<output.vocab_path>.test.csv`)

Example configs:

- `config/orb.yaml`
- `config/sift.yaml`

## Build (after yaml-cpp is installed)

`colcon build --packages-select fbow_vocab_trainer`

