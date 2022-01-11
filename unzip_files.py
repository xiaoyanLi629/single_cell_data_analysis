# pip install anndata
# pip install scipy
# pip3 install -U scikit-learn
# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
import zipfile

path_to_zip_file = 'output.zip'
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
	zip_ref.extractall('.')