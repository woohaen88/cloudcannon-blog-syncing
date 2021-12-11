---
date: 2021-12-11 00:00:00
title: GCP google instance에 GPU Driver 설치
description: 구글 GCP VM에서 GPU드라이버를 설치
tags:
image:
---
&nbsp;

**지원되는 운영체제**

해당 VM은 Ubuntu 18.04 LTS 버전에서 설치하였다.&nbsp;

구글 VM 할당 후 터미널에서 다음을 입력한다.

1\. Python 3이 운영체제에 설치되어 있는지 확인

2\. 설치 스크립트를 다운로드

\`\`\`curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install\_gpu\_driver.py --output install\_gpu\_driver.py\`\`\`

3\. 설치 스크립트를 실행

\`\`\`sudo python3 install\_gpu\_driver.py\`\`\`

설치 참조: [https://cloud.google.com/compute/docs/gpus/install-drivers-gpu](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)
