import numpy as np

from ImageClassifier import ImageClassifier
from sklearn.metrics.pairwise import cosine_similarity


class KMeans:
    def __init__(self):
        self.classifier = ImageClassifier()

    def image_cosine_similarity(self, image_url_1, image_url_2):
        vec_o_1 = self.classifier.run_inference_on_image(image_url_1)[0]
        vec_o_2 = self.classifier.run_inference_on_image(image_url_2)[0]
        print(cosine_similarity(vec_o_1, vec_o_2))

    def compute_centroid(self, image_urls):
        image_vectors = [self.classifier.run_inference_on_image(image_url)[0] for image_url in image_urls]
        centroid = np.mean(image_vectors, axis=0)
        return centroid


def main():
    shoe_url_1 = "https://images.viglink.com/product/250x250/images-nike-com/bc4e014d44579481f6b46ed658a690dc2c83de15.jpg?url=http%3A%2F%2Fimages.nike.com%2Fis%2Fimage%2FDotCom%2FPDP_HERO%2F881428_142_A_PREM%2Fair-jordan-12-retro-cny-big-kids-shoe.jpg"
    shoe_url_2 = "https://images.viglink.com/product/250x250/cdn-media-holabirdsports-com/c1f593ddb7fa1101d7d31b5a42ac3b9704a16232.jpg?url=http%3A%2F%2Fcdn.media.holabirdsports.com%2Fmedia%2Fcatalog%2Fproduct%2Fcache%2F1%2Fimage%2F260x260%2F17f82f742ffe127f42dca9de82fb58b1%2F0%2F4%2F041150_3.jpg"

    drone_url_1 = "https://images.viglink.com/product/250x250/images-na-ssl-images-amazon-com/b494dfd75a783bb80b7c96986383588f471b5662.jpg?url=https%3A%2F%2Fimages-na.ssl-images-amazon.com%2Fimages%2FI%2F31HgZMEzt1L.jpg"
    drone_url_2 = "https://images.viglink.com/product/250x250/ecx-images-amazon-com/68de10120e3253e9764d8798355e59333e3387a8.jpg?url=http%3A%2F%2Fecx.images-amazon.com%2Fimages%2FI%2F31nMC6JBaKL.jpg"

    incorrect_url="https://images.viglink.com/product/250x250/d1cr7zfsu1b8qs-cloudfront-net/c051917cc542a20edc8edbdcaefe2ac632f9b1bb.jpg?url=http%3A%2F%2Fd1cr7zfsu1b8qs.cloudfront.net%2Fassets%2Fimages%2Floading.gif"

    km = KMeans()
    #km.image_cosine_similarity(shoe_url_1, incorrect_url)
    # km.image_cosine_similarity(drone_url_1, drone_url_2)
    # km.image_cosine_similarity(drone_url_1, shoe_url_1)


    print(km.compute_centroid([shoe_url_1, shoe_url_2, drone_url_1, drone_url_2]))


if __name__ == "__main__":
    main()


