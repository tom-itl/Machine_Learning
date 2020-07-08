printf 'First image - Wild Pansy'
python predict.py ./test_images/wild_pansy.jpg flower_recog_model.h5
printf '\n\nSecond image - Orange Dahlia'
python predict.py ./test_images/orange_dahlia.jpg flower_recog_model.h5
printf '\n\nThird image - Hard-Leaved Pocket Orchid'
python predict.py ./test_images/hard-leaved_pocket_orchid.jpg flower_recog_model.h5
printf '\n\nFourth image - Cautleya Spicata'
python predict.py ./test_images/cautleya_spicata.jpg flower_recog_model.h5
