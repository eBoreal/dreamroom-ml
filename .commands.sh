python3 -m pip install -r requirements.txt


# curl https://get.docker.com | sh \
#   && sudo systemctl --now enable docker


# distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
#       && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
#       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
#             sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#             sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# sudo apt-get update
# sudo apt-get install -y nvidia-docker2


# # brev
# brev shell serverless-pix2pix
# brev open serverless-pix2pix --wait


eBoreal
HhaKCgqni2490

docker eboreal/pix2pix-serverless eboreal/pix2pix-serverless:v0
docker push eboreal/pix2pix-serverless:v0