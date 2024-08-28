HOST=conduit2.cs.uni-saarland.de
SRC=/home/jgross/kinodata-3D-affinity-prediction/data/ig_attributions
TARGET=./data/remote_ig_attrbibutions/
scp -r jgross@${HOST}:${SRC} ${TARGET}
