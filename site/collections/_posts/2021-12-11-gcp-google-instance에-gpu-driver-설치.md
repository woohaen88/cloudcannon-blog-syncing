---
date: 2021-12-11 00:00:00
title: GCP google instance에 GPU Driver 설치
description: 구글 GCP VM에서 GPU드라이버를 설치
tags:
  - gcp
  - GPU monitoring
image: /images/mount-google-storage-into-gcp-vm/google_cloud_logo.png
---

## 1. GCP-VM에 GPU 설치

**지원되는 운영체제**

해당 VM은 Ubuntu 18.04 LTS 버전에서 설치하였다. `google cloud console`에서 `ssh`를 눌러 `google cloud terminal`을 연다.

<figure style="text-align:center;"><img width="2940" height="512" src="/uploads/7.png" /><figcaption>Figure 1. Google Cloud Console</figcaption></figure>

SSH연결을 누르게 되면 아래와 같은 화면이 나오게 된다.

<figure style="text-align:center;"><img width="1800" height="512" src="/uploads/6.png" /><figcaption>Figure 2. Google Cloud Terminal 열린 모습</figcaption></figure>

구글 VM 할당 후 터미널에서 다음을 입력한다.

1. Python 3이 운영체제에 설치되어 있는지 확인
2. 설치 스크립트를 다운로드

```bash
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py –output install_gpu_driver.py
```


3. 설치 스크립트를 실행
```bash
sudo python3 install_gpu_driver.py
```

GPU 드라이버 설치가 잘 안된다면 아래 구글 공식문서를 참고한다.
설치 참조: [https://cloud.google.com/compute/docs/gpus/install-drivers-gpu](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)


만약에 `sudo nvidia-smi`를 입력했을 때
```txt
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```
와 같은 에러가 나타난다면 gpu 드라이버를 지웠다가 다시 설치하자.

```bash
sudo apt --installed list | grep nvidia-driver
nvidia-driver-{gpu 버전}/unknown,now 495.29.05-0ubuntu1 amd64 [installed,automatic]
sudo apt remove nvidia-driver-{gpu 버전}
sudo reboot now # 재부팅을 하지 않으면 적용되지 않는다.
```

<hr>

## 2. GPU 측정항목 모니터링 설정

기본적으로 **Google Cloud Platform**에는 다양한 리소스 모니터링을 제공하지만 **GPU 모니터링은 제공하지 않는다.** 따라서 GPU관련 모듈을 설치하고 `Google Cloud Platform`으로 연결한다.

기본적으로 공식문서를 따르면 문제 없을 것이다.
[GCP와 GPU 연결](https://cloud.google.com/compute/docs/gpus/monitor-gpus)

### 요구사항



각 VM에서 다음 요구사항이 충족되는지 확인합니다.

<!-- <blockqutoe/> -->
> * 각 VM에는 [GPU 연결](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus)완료된 상태여야 합니다.
> * 각 VM에는 [GPU 드라이버 설치](https://cloud.google.com/compute/docs/gpus/nstall-drivers-gpu#install-gpu-driver)가 완료된 상태여야 합니다.
> * 각 VM에는 Python 3.6 이상이 설치되어 있어야 합니다.
> * 각 VM에는 Python 가상 환경을 만드는 데 필요한 패키지가 있어야 합니다.


<br>

### 에이전트 다운로드

모니터링 스크립트를 /opt/google 디렉터리에 다운로드합니다. 다음 두 가지 기본 옵션(git, curl)이 있지만 여기서는 `git`을 사용하기로 한다.

```bash
# We need to use sudo to be able to write to /opt 
sudo mkdir -p /opt/google 
cd /opt/google 
sudo git clone https://github.com/GoogleCloudPlatform/compute-gpu-monitoring.git
```


### 가상 환경 설정

모니터링 스크립트를 사용하려면 필요한 모듈을 설치해야 합니다. 기본 Python 설치와 별개로 이 모듈에 대해 가상 환경을 만드는 것이 좋습니다.pipenv, virtualenv 2개중 하나를 이용할 수 있으나 여기서는 virtualenv를 사용한다.

virtualenv 및 pip를 사용하는 경우 가상 환경을 만들어야 하는데 환경을 만들려면 다음 명령어를 실행한다.

```bash
cd /opt/google/compute-gpu-monitoring/linux
sudo apt-get -y install python3-venv
sudo python3 -m venv venv
sudo venv/bin/pip install wheel sudo venv/bin/pip install -Ur requirements.txt
```

### 시스템 부팅 시 에이전트 시작

서비스 관리를 위해 systemd를 사용하여 시스템에서는 다음 단계를 수행하여 자동으로 시작되는 서비스 목록에 GPU 모니터링 에이전트를 추가한다.

`google_gpu_monitoring_agent_venv.service` 파일에는 virtualenv를 사용한 설치를 위해 준비된 systemd에 대한 서비스 정의가 포함되어 있다.

```bash
cd /opt/google/compute-gpu-monitoring/linux
sudo python3 -m venv venv
sudo venv/bin/pip install wheel
sudo venv/bin/pip install -Ur requirements.txt
```

## Cloud Monitoring에서 측정항목 검토

1. Google Cloud Console에서&nbsp;**측정항목 탐색기**&nbsp;페이지로 이동합니다.[Monitoring으로 이동](https://console.cloud.google.com/monitoring/metrics-explorer){: target="console"}
2. **리소스 유형**&nbsp;드롭다운에서&nbsp;**VM 인스턴스**를 선택합니다.

<figure style="text-align:center;"><img src="/uploads/3.png" /><figcaption>Figure 3. 리소스 유형 선택</figcaption></figure>

3. **측정항목**&nbsp;드롭다운에서 `custom/instance/gpu/utilization`을 입력합니다.
   
<figure style="text-align:center;"><img src="/uploads/4.png" /><figcaption>Figure 4. gpu utilization 입력</figcaption></figure>

<figure style="text-align:center;"><img src="/uploads/5.png" /><figcaption>Figure 5. gpu utilization 입력</figcaption></figure>





   **참고:**&nbsp;커스텀 측정항목이 표시되는 데 다소 시간이 걸릴 수 있으며, 다음 결과와 유사한 GPU 사용률이 나온다.

