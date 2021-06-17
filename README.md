# lec-docker
docker（主に機械学習に関する）についての学習用のリポジトリです。
<details>    
<summary>目次（クリックすると目次が表示されます）</summary>    
  
* [資料](#document)    
* [インストール手順](#install_method)
* [練習用コンテキスト](#pra_context)
* [dockerhubとgithubの連携](#dockerhub_github)
</details>

[](ここから資料--------------------------------------------------------)
<a id="document"></a>
# 資料
- [機械学習のためのdocker](https://github.com/cu-milab/lec-docker/blob/main/document/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AEdocker.pdf)    
初めて学習するひとはこちら   

- [機械学習のためのdocker_勉強会用](https://github.com/cu-milab/lec-docker/blob/main/document/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AEdocker_%E5%8B%89%E5%BC%B7%E4%BC%9A%E7%94%A8.pdf)    
勉強会用です。動画が埋め込んであるのでパワーポイント版を見ることをおすすめします。    

- [機械学習のためのdocker_簡易説明書](https://github.com/cu-milab/lec-docker/blob/main/document/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_docker_%E7%B0%A1%E6%98%93%E8%AA%AC%E6%98%8E%E6%9B%B8.pdf)    
最低限の説明のみ書いてあります。[機械学習のためのdocker](https://github.com/cu-milab/lec-docker/blob/main/document/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AEdocker.pdf)で一通りdockerについて学習した後は、こちらだけ見れば（機械学習については）大丈夫だと思います。

- [dockerhubとgithubの連携](https://github.com/cu-milab/lec-docker/blob/main/document/dockerhub%E3%81%A8github%E3%81%AE%E9%80%A3%E6%90%BA.pdf)    
dockerhubとgithubを連携させることで，dockerで構築した環境を管理しやすくなります．
（この資料は[機械学習のためのdocker](https://github.com/cu-milab/lec-docker/blob/main/document/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AEdocker.pdf)が理解できている前提になります． ）
[](ここまで資料--------------------------------------------------------)

[](ここからインストール手順-------------------------------------------------------------)
</br>     
<a id="install_method"></a>
# インストール手順
機械学習におけるdockerの導入イメージは下図になります．    
<img src="https://github.com/cu-milab/lec-docker/blob/main/document/fig/machine_learn_docker.jpg" width=600px>    
以降で，実際のインストール手順を説明します．
## 1. ubuntuにDockerEngineをインストール
**dockerは頻繁に更新されるため下記公式サイトを参考にしてインストールしたほうがよいです。**（英語のサイトですが手順どおりやればそんなに難しくないです）    
https://docs.docker.com/engine/install/ubuntu/
## 2. dockerをsudoなしで使用
dockerはデフォルトではsudoをつけて実行する必要がありますが、下記サイトを参考にすればsudo無しで実行することが可能です。    
https://qiita.com/DQNEO/items/da5df074c48b012152ee    
※ dockerの場合はsudo無しで使用されることが多いのでこの設定にしてもOKですが、一般的にsudoを無しで実行できてしまうとセキュリティ面など色々問題が起こる可能性があるので気をつけましょう。
## 3. NVIDIA driverをインストール
  ### 3-1. GPUを確認する
  下記コマンドにて自分のPCに搭載されているGPUを確認します。
  ```
  $  lspci | grep -i nvidia
  ```
  ### 3-2. NVIDIA driverを調べる
  調べたGPUに対応する「NVIDIA driverの最新ver（基本的には最新でOK）」を[ここから](https://www.nvidia.co.jp/Download/index.aspx?lang=jp)確認します。    

  ### 3-3. NVIDIA driverとCUDAの対応を調べる
  NVIDIA driverとCUDAの対応表を[ここ](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)より確認します。ここで3-1.で調べた「NVIDIA driverの最新ver」が「CUDAのどのver」までサポートしているかを確認します。ここで、使用したいCUDAにGPUが対応していない場合、GPUを新しいものに買い換える必要がある可能性があります。     
  例）CUDA9.0を使用したい場合：(linuxOSの場合)「NVIDIA driverの最新ver」が384.81以上ならばOK    

  ### 3-3. インストールファイルをダウンロード
  3-2.で条件が満たされていたら3-1.で調べた「NVIDIA driverの最新ver」のインストールファイルをダウンロードします。    

  ### 3-4. NVIDIA driverをインストール
  以下の手順により、3-3.でダウンロードしたファイルをインストールします。     

  実行ファイル化
  ```
  $ chmod +x <ダウンロードしたインストールファイル>
  ``` 

  ドライバインストール
  ```
  $ sudo ./＜ダウンロードしたインストールファイル＞
  ```

  再起動
  ```
  $ reboot
  ```   

  **ドライバがインストールされているか確認（正しくインストールされている場合は、下記コマンドが実行可能）。**
  ```
  $ nvidia-smi
  ```

  **「nvidia-dimが使用中」というエラーが出たら以降を実行する。**

  CUIモードに変更する。CUIモードになったら「alt+F1」→（PCの）ユーザ名、パスワードを入力してログインします。    
  ```
  $ systemctl isolate multi-user.target
  ```

  下記2つを無効化します。
  ```
  $ modprobe -r nvidia-drm
  ```
  ```
  $ modprobe -r nvidia-modeset
  ```

  ドライバインストール
  ```
  $ sudo ./＜ダウンロードしたファイル＞
  ```

  再起動
  ```
  $ reboot
  ```

  **ドライバがインストールされているか確認（正しくインストールされている場合は、下記コマンドが実行可能）。**
  ```
  $ nvidia-smi
  ```
## 4. nvidia-container-toolkitをインストール
**dockerは頻繁に更新されるため下記githubを参考にしてインストールしたほうがよいです。**（英語のサイトですが手順どおりやればそんなに難しくないです）     
https://github.com/NVIDIA/nvidia-docker  

## 参考サイト
- [NvidiaドライバとCUDAとcuDNNとTensorflow-gpuとPythonのバージョンの対応](https://qiita.com/konzo_/items/a6f2e8818e5e8fcdb896)        
- [Ubuntuの18.04 - NVIDIAドライバをインストールする方法](https://codechacha.com/ja/install-nvidia-driver-ubuntu/)    

[](ここまでインストール手順-------------------------------------------------------------)

[](ここから練習用コンテキスト------------------------------------------------------------)
</br>    
<a id="pra_context"></a>
# 練習用コンテキスト
Dockerfileのbuild～containerの生成までの流れのイメージ図は下図になります．    
<img src = "https://github.com/cu-milab/lec-docker/blob/main/document/fig/docker_method.jpg" width=600px>    
以降で，実際の手順を説明します．
## 1. Dockerfileをビルドしてdocker imageを生成
今回，使用するDockerfileは下図のようになります．    
<img src = "https://github.com/cu-milab/lec-docker/blob/main/document/fig/dockerfile.jpg" width=600px>    
</br>
Dockerfileをbuildするコマンド（`pra_docker_build_context`内で下記コマンドを実行）
- オプション    
  `-t`docker imageに任意の名称をつけられる）
```
$ docker build -t <Docker_imageの名称（任意）> <Dockerfileのある場所（相対パス）>
```

今回は、    
    docker imgage名（任意の名称）`cuda9.0cudnn7chainer5.3.0`、    
    Dockerfileの場所`.（カレントディレクトリ）`    
とするため下記になります。
```
$ docker build -t cuda9.0cudnn7chainer5.3.0 .
```
## 2. docker imageからcontainerを生成、container内に入る
docker imageからcontainerを作成、containerに入るコマンド    
- オプション    
   `-it`表示をキレイにする（詳細略）    
   `--gpus`PCのGPUをcontainerに適応。allを指定すると、PC内の全てのGPUを使用    
   `--rm`containerから抜けたら自動でcontainerを削除    
   `--name`containerに任意の名称をつける    
   `-v`指定したディレクトリをマウントさせる    
   ※ `--rm`を指定するとcontainerから抜ける時にcontainerが自動削除されるので、一般的には`--rm`と`--name`は併用しない（今回は練習用なので両方指定）
```
$ docker run -it --gpus all -it --rm --name==＜containerの名前（任意）＞ \    
-v ＜PC側のマウントしたいディレクトまでのパス（絶対パス）＞:＜container側のマウントしたいディレクトリまでのパス（ディレクトリがなかったら自動で生成される）＞ \    
＜1.で作成したdocker image名(またはID)＞ ＜container生成時に実行するコマンド＞
```

今回は、    
    container名（任意の名称）`cuda90cudnn7chainer530`、    
    PC側のマウントしたいディレクトリまでのパス`~/lec-docker/pra_docker_build_context/files`、    
    container側のマウントしたいディレクトリまでのパス（予め作成していないのでcontainer作成時に自動生成）`/files`、    
    1.で作成したdocker image名`cuda9.0cudnn7chainer5.3.0`、    
    container作成時に実行するコマンド`bash`    
とするため下記になります。    
```
$ docker run --gpus all -it --rm --name=cuda90cudnn7chainer530 -v ~/lec-docker/pra_docker_build_context/files:/files cuda9.0cudnn7chainer5.3.0 bash
```
## 3. container内でプログラム実行
  ### 3-1. container内にGPUが反映されているか確認 
  順番に進めていれば、container内に入っているはずなのでGPUが正しく認識されているかを確認するため下記を実行します。
  ```
  $ nvidia-smi
  ```
  **この際「コマンドが実行されない」「表示された内容がPC側で実行した内容と一致しない（CUDA versionにErrと表示されるなど）」場合は[インストール手順](#install_method)が正しく行われていない可能性があります。**
  
  ### 3-2. マウントしたディレクトリへ移動
  マウントしたディレクトリ（今回の場合、「/files」）に移動します。
  ```
  $ cd files
  ```
  
  ### 3-3. プログラムを実行
  サンプルプログラムを実行します。
  ```
  $ python3 pws_autoencoder4_hyoka_ari.py
  ```
  エラーが発生せずに学習が終了すればOKです。学習が終わればcontainer内の「/files」に「result」と「theme_hukugen_gosa_and_rri.png」が生成されます。**また、PC側の`~/lec-docker/pra_docker_build_context`内にも同様のファイルが出現しているはずです（逆に、`~/lec-docker/pra_docker_build_context`にファイルを作成した場合もcontainer内の`/files`にも同じファイルが出現するはずです。）**
  ### 3-4. containerから抜ける
  container内で下記コマンドを実行するとcontainerから抜けれます。    
  **今回は`--rm`オプションを使用しているため、containerから抜けるとcontainerが自動で削除されます。マウントしたディレクトリ（今回の場合`/files`）以外の場所にあるファイルは削除されてしまうので気をつけてください。**    
  ```
  $ exit
  ```
  ## 4. まとめ：機械学習におけるDockerを使用した環境構築の流れ
  環境構築の流れは下図になります．    
  <img src ="https://github.com/cu-milab/lec-docker/blob/main/document/fig/docker_method2.jpg" width=700px>
[](ここまで練習用コンテキスト------------------------------------------------------------)

[](ここからdockerhubとgithubに連携------------------------------------------------------------)
<a id="dockerhub_github"></a>
# dockerhubとgithubの連携
**以降の説明は，[dockerhubとgithubの連携](https://github.com/cu-milab/lec-docker/blob/main/document/dockerhub%E3%81%A8github%E3%81%AE%E9%80%A3%E6%90%BA.pdf)でdockerhubとgithubの連携の初期設定が済んでいる前提になります．**    
dockerhubとgithubの連携手順のイメージ図は以下になります．    
<img src = "https://github.com/cu-milab/lec-docker/blob/main/document/fig/dockerhub_github2.jpg" width=900px>    
以降で実際の手順を説明します．
  ### 1. Dockerfileの作成   
  Dockerfileを作成します．   
  ### 2. Dockerfileをbuild
  ホスト側でDockerfileをbuild（正しくbuildができるか確認）します．    
  ```
  $ docker build .
  ```
  ### 3. githubへpush    
  2.でbuildしたDockerfileが入ったリポジトリをgithubへpushします（latestを更新）
  ```
  $ git add .
  ```
  ```
  $ git commit -m “コメント"
  ```
  ```
  $ git push origin master
  ```
  ### 4. タグ名を指定して再push（バージョン管理）   
  3.でpushしたリポジトリをタグ名を指定して再pushします（バージョン管理） 
  ```
  $ git tag -a ＜タグ（タグ名は「1.0」等の数値のみ）＞ -m “コメント"
  ```
  ```
  $ git push origin ＜タグ（上記コマンドで設定したタグ名）＞
  ```
  ### 5. リポジトリの確認
  dockerhub内のリポジトリを確認して    
  - 3.でpushした結果，タグ名がlatestのdocker imageが最新に更新されている    
  - 4.でpushした結果，指定したタグ名のdocker imageがアップロードされている   
  
  ことを確認してください．    
  </br>
上記の連携は下図のように，各プロジェクトごとに行ったほうが管理がしやすいです．    
<img src="https://github.com/cu-milab/lec-docker/blob/main/document/fig/dockerhub_github1.jpg" width=600px>    

[](ここまでdockerhubとgithubに連携------------------------------------------------------------)
