---
date: 2021-12-11 00:00:00
title: GCP google instance에 GPU Driver 설치
description: 구글 GCP VM에서 GPU드라이버를 설치
tags:
  - gcp
  - GPU monitoring
image: /uploads/gcp.png
---
&nbsp;

**지원되는 운영체제**

해당 VM은 Ubuntu 18.04 LTS 버전에서 설치하였다.&nbsp;

구글 VM 할당 후 터미널에서 다음을 입력한다.

1\. Python 3이 운영체제에 설치되어 있는지 확인

2\. 설치 스크립트를 다운로드

`curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install\_gpu\_driver.py –output install\_gpu\_driver.py`

3\. 설치 스크립트를 실행

`sudo python3 install\_gpu\_driver.py`

![](/uploads/7.png){: width="2940" height="512"}

&nbsp;

![](/uploads/6.png){: width="1800" height="1412"}

설치 참조: [https://cloud.google.com/compute/docs/gpus/install-drivers-gpu](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)

&nbsp;

에러: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

```bash
sudo apt --installed list | grep nvidia-driver
nvidia-driver-495/unknown,now 495.29.05-0ubuntu1 amd64 [installed,automatic]
sudo apt remove nvidia-driver-49
sudo reboot now
```

&nbsp;

## GPU 측정항목 보고 스크립트 설정

### 요구사항

각 VM에서 다음 요구사항이 충족되는지 확인합니다.

* 각 VM에는&nbsp;[GPU 연결](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus)이 완료된 상태여야 합니다.
* 각 VM에는&nbsp;[GPU 드라이버 설치](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#install-gpu-driver)가 완료된 상태여야 합니다.
* 각 VM에는 Python 3.6 이상이 설치되어 있어야 합니다.
* 각 VM에는 Python 가상 환경을 만드는 데 필요한 패키지가 있어야 합니다.

&nbsp;

### 에이전트 다운로드

모니터링 스크립트를 /opt/google 디렉터리에 다운로드합니다. 다음 두 가지 기본 옵션이 있습니다.

* git 유틸리티를 사용하여 다운로드
* curl을 사용하여 패키지로 다운로드

[git 사용](https://cloud.google.com/compute/docs/gpus/monitor-gpus#git-%EC%82%AC%EC%9A%A9)[ZIP 패키지로 다운로드](https://cloud.google.com/compute/docs/gpus/monitor-gpus#zip-%ED%8C%A8%ED%82%A4%EC%A7%80%EB%A1%9C-%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C)

\# We need to use sudo to be able to write to /opt sudo mkdir -p /opt/google cd /opt/google sudo git clone https://github.com/GoogleCloudPlatform/compute-gpu-monitoring.git

### 가상 환경 설정

모니터링 스크립트를 사용하려면 필요한 모듈을 설치해야 합니다. 기본 Python 설치와 별개로 이 모듈에 대해 가상 환경을 만드는 것이 좋습니다. 이 가상 환경을 만들려면 pipenv 또는 virtualenv를 사용합니다.

[virtualenv 사용](https://cloud.google.com/compute/docs/gpus/monitor-gpus#virtualenv-%EC%82%AC%EC%9A%A9)[pipenv 사용](https://cloud.google.com/compute/docs/gpus/monitor-gpus#pipenv-%EC%82%AC%EC%9A%A9)

virtualenv 및 pip를 사용하는 경우 가상 환경을 만들어야 합니다. 환경을 만들려면 다음 명령어를 실행합니다.

```bash
cd /opt/google/compute-gpu-monitoring/linux
sudo apt-get -y install python3-venv
sudo python3 -m venv venv
sudo venv/bin/pip install wheel sudo venv/bin/pip install -Ur requirements.txt
```

### 시스템 부팅 시 에이전트 시작

서비스 관리를 위해 systemd를 사용하여 시스템에서는 다음 단계를 수행하여 자동으로 시작되는 서비스 목록에 GPU 모니터링 에이전트를 추가합니다.

[virtualenv 사용](https://cloud.google.com/compute/docs/gpus/monitor-gpus#virtualenv-%EC%82%AC%EC%9A%A9)[pipenv 사용](https://cloud.google.com/compute/docs/gpus/monitor-gpus#pipenv-%EC%82%AC%EC%9A%A9)

google\_gpu\_monitoring\_agent\_venv.service 파일에는 virtualenv를 사용한 설치를 위해 준비된 systemd에 대한 서비스 정의가 포함되어 있습니다.

```bash
cd /opt/google/compute-gpu-monitoring/linux
sudo python3 -m venv venv
sudo venv/bin/pip install wheel
sudo venv/bin/pip install -Ur requirements.txt
```

## &nbsp;

## &nbsp;

## ![](/uploads/3.png){: width="4096" height="2560"}

&nbsp;

![](/uploads/4.png){: width="4096" height="2560"}

&nbsp;

![](/uploads/5.png){: width="4096" height="2560"}

&nbsp;

&nbsp;

&nbsp;

## Cloud Monitoring에서 측정항목 검토

1. Google Cloud Console에서&nbsp;**측정항목 탐색기**&nbsp;페이지로 이동합니다.[Monitoring으로 이동](https://console.cloud.google.com/monitoring/metrics-explorer){: target="console"}
2. **리소스 유형**&nbsp;드롭다운에서&nbsp;**VM 인스턴스**를 선택합니다.
3. **측정항목**&nbsp;드롭다운에서 custom/instance/gpu/utilization을 입력합니다.

   **참고:**&nbsp;커스텀 측정항목이 표시되는 데 다소 시간이 걸릴 수 있습니다. 다음 결과와 유사한 GPU 사용률이 나옵니다.

## 다음 단계

* [Compute Engine의 GPU](https://cloud.google.com/compute/docs/gpus)&nbsp;자세히 알아보기
* GPU 호스트 유지보수를 처리하려면&nbsp;[GPU 호스트 유지보수 이벤트 처리](https://cloud.google.com/compute/docs/gpus/gpu-host-maintenance)를 참조하세요.
* GPU 성능을 최적화하려면&nbsp;[GPU 성능 최적화](https://cloud.google.com/compute/docs/gpus/optimize-gpus)를 참조하세요.
