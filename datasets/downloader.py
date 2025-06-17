import subprocess, os

class Downloader():
    CITYSCAPES_LINK = [
        "1uINto6zVV0VY9h380Pnprt_PR0izLfzW",
        "1O8VGxrdRgRCt8lw36SXFNPIvfYnDZvOs",
        "1lCo34lp6BpIvUSwmKO7xbp5hGuXywGFs",
        "1d4rzU3aExnuiooNaAr0miZ_gZmWAEH1x",
        "1lY1VGuE0UKsUGTFNNQf7YweolGzuJoRM",
        '1o09HNrDi2rNrgZXmfHngGliuqT6xlvDQ',
        '1sJ85KIaBkjN_6SiXPcsJYqaxfrjrLgX_',
        '1_0N0415q9cwV9uDKPXbXvJKrretduhph',
        '1TtOYU9OkLoy8Hs3FItMFnhYFtWJNT2wM',
        '1Q0wB-kIQBF6wivedP-Mac3CswKmC4b5-',
        '1u3kA7gZ-GtNbPSRuFtfoUDeQxyTZvLpt'
    ]
    
    GTAV_LINK = [
        '1wTgWtyfG415XQuJyOxK6xbT52VVarJO7',
        '1qyBW0Weh0D4LltxPn_-KA9aWiiYRKdBW',
        '1oDRTmXacw23ksYrjqG8WpPUB-2I7_mb1',
        '1gniMsHTZ5i2L4qjAwrsH5C9394ZzlLgu',
        '1wYvdtQ4dUDeZLp4LF0-PLoxZV-opAiQM',
        '1fu7Z0DSSJLFQEBx2E8MFfq3sz1TCtwwb',
        '1sJ85KIaBkjN_6SiXPcsJYqaxfrjrLgX_',
        '1oxDuTMHCJjhpLh11Oq4G9WUyhU7hVkJ6',
        '1gI6P62DWuBvIYCFznO8vFlXjmLJaO47g',
        '1PxzzSi4W7fPK7dpxhjWKaqOepdK-p5jC',
        '1xQt8EfoAbflIzMcanJHLlaQB5YgTCUg6'
    ]
    
    DEEPLAB_WEIGHTS_LINK = [   
        '19bnz8U9YILyLjOTFuzeGhHRrYheW4bml',
        '1YG7ynbwSmyVyMOJ17ePfaGYDj6Xp6dq2'
    ]
    
    def downloadCityScapes(self)->bool:
        """
        Downloads the CityScapes dataset from the stored link.
        
        Returns:
            status (bool): True if the download was successful, False otherwise.
        """
        flag = False
        for link in self.CITYSCAPES_LINK:
            try:
                subprocess.run(["gdown", "--quiet", link, "-O", "CityScapes.zip"],check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error downloading CityScapes dataset: {e} with id {link}")
                continue

            if 'CityScapes.zip' in subprocess.check_output("ls", shell=True).decode('utf-8').split():
                flag = True
                break
        
        if not flag:
            print("CityScapes dataset download failed.")
            return False

        # unzip the dataset
        try:
            subprocess.run(["unzip", "-q", "CityScapes.zip", "-d", "data"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error unzipping CityScapes dataset: {e}")
            return False

        # remove the zip used for the download
        try:
            subprocess.run(["rm", "CityScapes.zip"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error removing CityScapes zip file: {e}")
            return False

        return True
    
    
    def downloadGTA5(self)->bool:
        """
        Downloads the GTA5 dataset from the stored link.
        
        Returns:
            status (bool): True if the download was successful, False otherwise.
        """
        flag = False
        for link in self.GTAV_LINK:
            try:
                subprocess.run(["gdown", "--quiet", link, "-O", "GTA5.zip"],check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error downloading GTA5 dataset: {e} with id {link}")
                continue

            if 'GTA5.zip' in subprocess.check_output("ls", shell=True).decode('utf-8').split():
                flag = True
                break
        
        if not flag:
            print("GTA5 dataset download failed.")
            return False

        # unzip the dataset
        try:
            subprocess.run(["unzip", "-q", "GTA5.zip", "-d", "data"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error unzipping GTA5 dataset: {e}")
            return False
        
        # remove the zip used for the download
        try:
            subprocess.run(["rm", "GTA5.zip"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error removing GTA5 zip file: {e}")
            return False
        return True
    
    
    def downloadWeightsDeepLabV2(self)->bool:
        """
        Downloads the DeepLabV2 weights from the stored link.
        
        Returns:
            status (bool): True if the download was successful, False otherwise.
        """
        flag = False
        if 'weights' not in subprocess.check_output("ls", shell=True).decode('utf-8').split():
            subprocess.run(["mkdir", "weights"], check=True)

        if 'DeepLabV2' not in os.listdir('weights'):
            os.mkdir('weights/DeepLabV2')

        for link in self.DEEPLAB_WEIGHTS_LINK:
            try:
                subprocess.run(["gdown", "--quiet", link, "-O", "weights/DeepLabV2/weights_0_0.pth"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error downloading DeepLabV2 weights: {e} with id {link}")
                continue

            if 'weights/DeepLabV2/weights_0_0.pth' in subprocess.check_output("ls", shell=True).decode('utf-8').split():
                flag = True
                break
            
        if not flag:
            print("DeepLabV2 weights download failed.")
            return False
        
        return True