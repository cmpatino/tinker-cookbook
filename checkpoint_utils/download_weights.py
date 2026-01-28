import tinker
import urllib.request
 
sc = tinker.ServiceClient(api_key="tml-SfciZtUbPhedxhlIMWYKLidG0CRl2qJikDFlvfyMfSBapFgpu7Q6artD5TpBUfk4KAAAA")
rc = sc.create_rest_client()
future = rc.get_checkpoint_archive_url_from_tinker_path("tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final")
checkpoint_archive_url_response = future.result()
 
# `checkpoint_archive_url_response.url` is a signed URL that can be downloaded
# until checkpoint_archive_url_response.expires
urllib.request.urlretrieve(checkpoint_archive_url_response.url, "sft-r128.tar")