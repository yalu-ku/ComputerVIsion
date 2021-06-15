from google_images_download import google_images_download


def imageCrwaling(keyword, dir):
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": keyword,
                 "limit": 1000,
                 "chromedriver": "chromedriver",
                 "print_urls": False,
                 "no_directory": True,
                 "output_directory": dir}
    paths = response.download(arguments)
    print(paths)


# imageCrwaling('maltese', 'datasets/maltese')
imageCrwaling('말티즈', 'datasets/maltese')
