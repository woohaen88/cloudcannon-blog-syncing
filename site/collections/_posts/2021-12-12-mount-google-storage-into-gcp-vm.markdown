---
date: 2021-12-11 00:00:00
title: GCP VM에 Google Cloud Storage Mount
description: GCP VM에 Google Cloud Storage Mount하기
tags:
- gcp
- google cloud storage
image: /images/mount-google-storage-into-gcp-vm/google_cloud_logo.png
---

<!-- 포스트 이미지 폴더: /images/mount-google-storage-into-gcp-vm/ -->
<h1>Google Compute Engine 에 Cloud Storage 마운트하기</h1>

<ol>
  <li>Cloud Storage bucket</li>
  <li>Cloud Engine</li>
  <li>gcsfuse</li>
</ol>

<h2>1. Compute Engine 만들기</h2>
Cloude Storage Bucket을 마운트하기 위해서는 gcsfuse가 필요하다.

<blockquote>
  <em>Cloud Storage FUSE는 Cloud Storage 버킷을 Linux 또는 macOS 시스템에 파일 시스템으로 마운트할 수 있는 오픈소스 FUSE 어댑터입니다. Google Compute Engine VM 또는 온프레미스 시스템1을 포함하여 Cloud Storage와 연결된 모든 위치에서 Cloud Storage FUSE를 실행할 수 있습니다.
  <a
    href="https://cloud.google.com/storage/docs/gcs-fuse#using_feat_name">https://cloud.google.com/storage/docs/gcs-fuse#using_feat_name</a>  
  </em>
</blockquote>

먼저 Google Cloud VM에 접속한다.

1. gcsfuse 배포 URL을 패키지 소스로 추가하고 공개 키를 가져온다.
```bash
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
```

2. 사용 가능한 패키지 목록을 업데이트 하고 gcsfuse를 설치한다.
```bash
sudo apt-get update
sudo apt-get install gcsfuse
```

3. gcsfuse에 대한 향후 업데이트는 다음과 같은 일반적인 방법으로 설치할 수 있다. 
```bash
sudo apt-get update && sudo apt-get -y upgrade
```

<hr/>

<h2>2. 버킷 마운트하기</h2>
<h3>사전준비</h3>
처음에 google-cloud와 로그인 인증을 해준다.
```bash
gcloud auth login
```
크롬 브라우저에서 로그인 설정을 해준다. 이게 완료되어야 다음단계로 진행된다.
<h3>버킷마운트</h3>
```bash
mkdir -p /path/to/mount/point
gcsfuse {google_cloud_bucket} /path/to/mount/point --implicit-dirs
```

gcsfuse 를 이용하면 google cloud storage 를 파일시스템으로 mount를 시킬 수 있다. 이 때 중요한 것은 --implicit-dirs 옵션을 붙여주여야 디렉토리 구조가 보인다는 것이다.

언마운트는 다음과 같이 진행한다.
```bash
fusermount -u /path/to/mount/point
```