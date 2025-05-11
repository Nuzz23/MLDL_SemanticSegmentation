import subprocess

class Downloader():
    CITYSCAPES_LINK = [
        "1uINto6zVV0VY9h380Pnprt_PR0izLfzW",
        "1O8VGxrdRgRCt8lw36SXFNPIvfYnDZvOs",
        "1lCo34lp6BpIvUSwmKO7xbp5hGuXywGFs",
        "1d4rzU3aExnuiooNaAr0miZ_gZmWAEH1x",
        "1lY1VGuE0UKsUGTFNNQf7YweolGzuJoRM",
        '1o09HNrDi2rNrgZXmfHngGliuqT6xlvDQ'
    ]
    
    GTAV_LINK = [
        '1wTgWtyfG415XQuJyOxK6xbT52VVarJO7',
        '1qyBW0Weh0D4LltxPn_-KA9aWiiYRKdBW',
        '1oDRTmXacw23ksYrjqG8WpPUB-2I7_mb1',
        '1gniMsHTZ5i2L4qjAwrsH5C9394ZzlLgu',
        '1wYvdtQ4dUDeZLp4LF0-PLoxZV-opAiQM',
        '1fu7Z0DSSJLFQEBx2E8MFfq3sz1TCtwwb'
    ]
    
    
    def downloadCityScapes(self)->bool:
        """
        Downloads the CityScapes dataset from the stored link.
        
        Returns:
            status (bool): True if the download was successful, False otherwise.
        """
        flag = False
        for link in self.CITYSCAPES_LINK:
            subprocess.run(["gdown", "--quiet", link, "-O", "CityScapes.zip"],check=True)

            if 'CityScapes.zip' in subprocess.check_output("ls", shell=True).decode('utf-8').split():
                flag = True
                break
        
        if not flag:
            print("CityScapes dataset download failed.")
            return False

        # unzip the dataset
        subprocess.run(["unzip", "-q", "CityScapes.zip", "-d", "data"], check=True)

        # remove the zip used for the download
        subprocess.run(["rm", "CityScapes.zip"], check=True)

        return True
    
    
    
    def downloadGTA5(self)->bool:
        """
        Downloads the GTA5 dataset from the stored link.
        
        Returns:
            status (bool): True if the download was successful, False otherwise.
        """
        flag = False
        for link in self.GTAV_LINK:
            subprocess.run(["gdown", "--quiet", link, "-O", "GTA5.zip"],check=True)

            if 'GTA5.zip' in subprocess.check_output("ls", shell=True).decode('utf-8').split():
                flag = True
                break
        
        if not flag:
            print("GTA5 dataset download failed.")
            return False

        # unzip the dataset
        subprocess.run(["unzip", "-q", "GTA5.zip", "-d", "data"], check=True)

        # remove the zip used for the download
        subprocess.run(["rm", "GTA5.zip"], check=True)

        return True