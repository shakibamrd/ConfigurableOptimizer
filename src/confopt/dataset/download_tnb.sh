# This script downloads and renames all the taskonomydata_mini data for TransNAS-Bench-101
# Call the script from the root of the repository

datasets=(wainscott tolstoy klickitat pinesdale stockman beechwood coffeen corozal \
benevolence eagan forkland hanson hiteman ihlen lakeville lindenwood \
marstons merom newfields pomaria shelbyville uvalda)

# Creates main data folder if it does not exist.
if [ ! -d "datasets/taskonomydata_mini" ]; then
    mkdir -p "datasets/taskonomydata_mini"
fi
cd datasets/taskonomydata_mini

# Creates a subdirectory for each building if it does not exist.
for building in "${datasets[@]}"; do
    if [ ! -d "$building" ]; then
        mkdir -p "$building"
        echo "Folder created: $building"
    fi
done

echo "Folder structure checked successfully"

# download all rgb files
echo Checking for rgb files
for dataset in ${datasets[@]}
do
    file=$dataset\_rgb.tar
    filepath=http://downloads.cs.stanford.edu/downloads/taskonomy_data/rgb/$file
    cd $dataset
    if [ ! -d "rgb" ]
    then
        echo rgb data does not exist for $dataset
        wget $filepath
        tar -xvf $file
        rm $file
    fi
    cd ..
done

# download all class_object files
echo Checking for class_object files
for dataset in ${datasets[@]}
do
    file=$dataset\_class_object.tar
    filepath=http://downloads.cs.stanford.edu/downloads/taskonomy_data/class_object/$file
    cd $dataset
    if [ ! -d "class_object" ]
    then
        echo class_object data does not exist for $dataset
        wget $filepath
        tar -xvf $file
        rm $file
    fi
    cd ..
done

# download all class_scene files
echo Checking for class_scene files
for dataset in ${datasets[@]}
do
    file=$dataset\_class_scene.tar
    filepath=http://downloads.cs.stanford.edu/downloads/taskonomy_data/class_scene/$file
    cd $dataset
    if [ ! -d "class_scene" ]
    then
        echo class_scene data does not exist for $dataset
        wget $filepath
        tar -xvf $file
        rm $file
    fi
    cd ..
done

# rename all class_places.npy to class_scene.npy
for dataset in ${datasets[@]}
do
    for j in $dataset/class_scene/*class_places.npy
    do
        if [ -f $j ]; then
            echo Renaming $j 
            mv -- "$j" "${j%class_places.npy}class_scene.npy"
        fi
    done
done

# download all normal files
echo Checking for normal files
for dataset in ${datasets[@]}
do
    file=$dataset\_normal.tar
    filepath=http://downloads.cs.stanford.edu/downloads/taskonomy_data/normal/$file
    cd $dataset
    if [ ! -d "normal" ]
    then
        echo normal data does not exist for $dataset
        wget $filepath
        tar -xvf $file
        rm $file
    fi
    cd ..
done

echo Taskonomy_mini data for TransNASBench-101 downloaded successfully