#!/bin/bash -e

white_list="^cmake\|^third_party"
error_num=0
echo "check_result:" > /tmp/check_result_${USER}
for file in `git diff HEAD^ --name-only | grep "\.cpp$\|\.h$" | grep -v $white_list`;
do
    if [ ! -f $file ];then
        continue
    fi
    base_name_file=$(basename $file)
    clang-format-5.0 $file > /tmp/$base_name_file
    ret="`diff -aur $file /tmp/$base_name_file`"
    if [ $? != 0 ]; then
        echo [check falsed] $file >> /tmp/check_result_${USER}
        echo $file check falsed
        echo "diff result:\n"
        echo "`diff -aur $file /tmp/$base_name_file`"
        error_num=1
    fi
    rm /tmp/$base_name_file
done
if [ $error_num -eq 0 ];then
    echo "all files changed passed format check"
else
    cat /tmp/check_result_${USER}
fi
rm /tmp/check_result_${USER}
exit $error_num
