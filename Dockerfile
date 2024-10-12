FROM ghcr.io/pytorch/pytorch-nightly:2.3.0.dev20240305-cuda12.1-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output /opt/app/nnUNet/nnUNet_results /opt/app/code/ \
    && chown -R user:user /opt/app /input /output /opt/app/nnUNet/ /opt/app/code/

COPY --chown=user:user Dataset824_AortaSeg24_CTA_50 /opt/app/nnUNet/nnUNet_results/Dataset824_AortaSeg24_CTA_50
COPY --chown=user:user Dataset825_AortaSeg24_CTA_bin_50 /opt/app/nnUNet/nnUNet_results/Dataset825_AortaSeg24_CTA_bin_50
COPY --chown=user:user nnUNet_hti_cbdc_nnUNet_v2.5 /opt/app/code/nnUNet_hti_cbdc_nnUNet_v2.5

RUN cd /opt/app/code/nnUNet_hti_cbdc_nnUNet_v2.5 && pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install monai -i https://pypi.tuna.tsinghua.edu.cn/simple/

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user inference.py /opt/app/

# Entrypoint to your python code - executes process.py as a script
ENTRYPOINT ["python", "inference.py"]
