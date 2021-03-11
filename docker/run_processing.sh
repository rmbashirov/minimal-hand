#!/usr/bin/env bash


bsub -q normal -gpu "num=1:mode=shared" -oo $PWD/log_azure_test.txt -m airugpub01 \
    'export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') && \
    source ./source.sh && \
    docker pull $HEAD_NAME && \
    nvidia-docker run -t $PARAMS $VOLUMES $HEAD_NAME bash process_azure_people.sh 04_simple,04_complex,05_simple,05_complex 000062692912,000667292912'



#01_02_03
#04_05_06_07_09_10_11_14_15
#16_17_18_20_21_22_23_24_26
#27_28_30_31_32_33_34_35_36
#37_38_40_41_42_43_45_46_47
#48_49_50_51_52_54_55_56_57
#58_59_60_63_64_65_66_67

#bsub -q normal -gpu "num=1:mode=shared" -oo $PWD/log_01.txt -m airugpua02 \
#    'export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') && \
#    source ./source.sh && \
#    docker pull $HEAD_NAME && \
#    nvidia-docker run -t $PARAMS $VOLUMES $HEAD_NAME bash process_azure_people.sh 01 000583592412_000062692912_000667292912_000230292412'


#bsub -q normal -gpu "num=1:mode=shared" -oo $PWD/log_02.txt -m airugpua02 \
#    'export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') && \
#    source ./source.sh && \
#    docker pull $HEAD_NAME && \
#    nvidia-docker run -t $PARAMS $VOLUMES $HEAD_NAME bash process_azure_people.sh 02 000583592412_000062692912_000667292912_000230292412'
#
#
#bsub -q normal -gpu "num=1:mode=shared" -oo $PWD/log_03.txt -m airugpua02 \
#    'export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') && \
#    source ./source.sh && \
#    docker pull $HEAD_NAME && \
#    nvidia-docker run -t $PARAMS $VOLUMES $HEAD_NAME bash process_azure_people.sh 03 000583592412_000062692912_000667292912_000230292412'


#bsub -q normal -gpu "num=1:mode=shared:j_exclusive=yes" -oo $PWD/log_04_05_06_07_09_10_11_14_15.txt -m airugpub01 \
#    'export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') && \
#    source ./source.sh && \
#    docker pull $HEAD_NAME && \
#    nvidia-docker run -t $PARAMS $VOLUMES $HEAD_NAME bash process_azure_people.sh 04_05_06_07_09_10_11_14_15 000583592412_000062692912_000667292912_000230292412'
#
#
#bsub -q normal -gpu "num=1:mode=shared:j_exclusive=yes" -oo $PWD/log_16_17_18_20_21_22_23_24_26.txt -m airugpub01 \
#    'export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') && \
#    source ./source.sh && \
#    docker pull $HEAD_NAME && \
#    nvidia-docker run -t $PARAMS $VOLUMES $HEAD_NAME bash process_azure_people.sh 16_17_18_20_21_22_23_24_26 000583592412_000062692912_000667292912_000230292412'
#
#
#bsub -q normal -gpu "num=1:mode=shared:j_exclusive=yes" -oo $PWD/log_27_28_30_31_32_33_34_35_36.txt -m airugpub01 \
#    'export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') && \
#    source ./source.sh && \
#    docker pull $HEAD_NAME && \
#    nvidia-docker run -t $PARAMS $VOLUMES $HEAD_NAME bash process_azure_people.sh 27_28_30_31_32_33_34_35_36 000583592412_000062692912_000667292912_000230292412'


#bsub -q normal -gpu "num=1:mode=shared:j_exclusive=yes" -oo $PWD/log_37_38_40_41_42_43_45_46_47.txt -m airugpub03 \
#    'export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') && \
#    source ./source.sh && \
#    docker pull $HEAD_NAME && \
#    nvidia-docker run -t $PARAMS $VOLUMES $HEAD_NAME bash process_azure_people.sh 37_38_40_41_42_43_45_46_47 000583592412_000062692912_000667292912_000230292412'
#
#
#bsub -q normal -gpu "num=1:mode=shared:j_exclusive=yes" -oo $PWD/log_48_49_50_51_52_54_55_56_57.txt -m airugpub03 \
#    'export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') && \
#    source ./source.sh && \
#    docker pull $HEAD_NAME && \
#    nvidia-docker run -t $PARAMS $VOLUMES $HEAD_NAME bash process_azure_people.sh 48_49_50_51_52_54_55_56_57 000583592412_000062692912_000667292912_000230292412'


#bsub -q normal -gpu "num=1:mode=shared:j_exclusive=yes" -oo $PWD/log_58_59_60_63_64_65_66_67.txt -m airugpub01 \
#    'export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') && \
#    source ./source.sh && \
#    docker pull $HEAD_NAME && \
#    nvidia-docker run -t $PARAMS $VOLUMES $HEAD_NAME bash process_azure_people.sh 58_59_60_63_64_65_66_67 000583592412_000062692912_000667292912_000230292412'

