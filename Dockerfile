#FROM ベースとなるimageファイル（主にubuntuOS関連）
#docker hubの「nvedia/cuda」リポジトリを検索。Tagsから使用したいCudaのバージョンのimageを持ってくる
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

#RUN コマンドを実行
#必要なパッケージをインストール
#apt-get installの際に「-y」を指定することで、ターミナルに表示されるインタラクティブな入力を全て「y」で返答
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    vim

#WORKDIR RUN等を実行するディレクトリを変更
#注意
#下記のような場合でも「hoge」というディレクトリは「/opt」ではなく、「/」に生成されるため注意
#WORKDIR /
#RUN cd /opt
#RUN mkdir hoge
#ただし、
#RUN cd /opt && mkdir hoge
#のように&&で２つのコマンドを同時に実行した場合はちゃんと「/opt」にディレクトリが作成される
WORKDIR /opt

#anacondaのインストール
#オプションについて-------------------------------------
#b:バッチモード
#通常は、インストール途中にターミナルに「Please answer 'yes' or 'no':」のような入力を要求されることがあるが、バッチモードでインストールを開始すると、それが不要になる
#p:インストール先を変更
#anacondaの場合、デフォルトではroot直下に「anaconda3」というディレクトリが生成されるが、-pを指定すると任意の位置にanacondaをインストール可能
#※ このオプションは使用する.shファイルによって違うので注意（shファイルの中にオプションの設定が記述されているため）
#使用する.shファイルにどのようなオプションが存在するかは
#$ sh -x <sh_file>.sh   で確認可能
#-----------------------------------------------------------
#Dockerfileをbuildする際は、キーボード入力ができないため、上記のようなオプションを使用してキーボード入力が不要になるようにする必要がある
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh && \
    sh Anaconda3-2020.02-Linux-x86_64.sh -b -p /opt/anaconda3 && \
    rm -f Anaconda3-2020.02-Linux-x86_64.sh

#ENV パスを通す
ENV PATH /opt/anaconda3/bin:$PATH

#cupy→chainerの順番でインストールしないと依存関係の問題でうまく行かないため注意
RUN pip install --upgrade pip && pip install \
    #keras==2.3 \
    #scipy==1.4.1 \
    #tensorflow-gpu==2.1 \
    #torch==1.1.0 \
    #torchvision==0.3.0 \
    cupy-cuda90==5.3.0\
    chainer==5.3.0
WORKDIR /

#CMD $docker run の際にデフォルトで実行されるコマンド
CMD ["/bin/bash"]
