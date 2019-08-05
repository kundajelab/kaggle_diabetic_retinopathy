
for firstnum in 1 2 3; do
    for secondnum in 1 2 3 4; do
        #unzip Base"$firstnum""$secondnum".zip -d Base"$firstnum""$secondnum"
        #wget https://www.ceos-systems.com/file-sharing/Base"$firstnum""$secondnum".zip -O Base"$firstnum""$secondnum".zip
        #wget https://www.ceos-systems.com/file-sharing/Annotation_Base"$firstnum""$secondnum".xls -O Annotation_Base"$firstnum""$secondnum".xls
        cd Base"$firstnum""$secondnum"
        ln -s ../ds_images.py .
        python ds_images.py 2
        cd ..
    done
done
