import numpy as np
import torch as T
from torchvision import transforms
from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.utils as vutils
from src.datagen.scm_datagen import SCMDataGenerator
from src.datagen.scm_datagen import SCMDataTypes as sdt
from src.scm.scm import check_equal
from src.ds import CTF, CTFTerm
from src.metric.visualization import show_image_grid


def expand_do(val, n):
    return np.ones(n, dtype=int) * val


class ColorMNISTDataGenerator(SCMDataGenerator):
    def __init__(self, image_size, mode, repr_dim=None, normalize=False, evaluating=False):
        super().__init__(mode)

        self.evaluating = evaluating
        self.raw_mnist_n = 0
        self.raw_mnist_images = None
        self.normalize = normalize
        if not evaluating:
            mnist_data = MNIST('dat/mnist/')
            #mnist_data = MNIST('../../dat/mnist/')
            images, labels = mnist_data.load_training()
            self.raw_mnist_n = len(images)
            images = np.array(images).reshape((self.raw_mnist_n, 28, 28))
            labels = np.array(labels)

            self.raw_mnist_images = dict()
            for i in range(len(labels)):
                if labels[i] not in self.raw_mnist_images:
                    self.raw_mnist_images[labels[i]] = []
                self.raw_mnist_images[labels[i]].append(images[i])

        dark_red = (0.6, 0.0, 0.0)
        light_red = (1.0, 0.4, 0.4)
        dark_blue = (0.0, 0.0, 0.6)
        light_blue = (0.4, 0.4, 1.0)

        self.colors = {
            0: dark_blue,
            1: light_blue,
            2: light_red,
            3: dark_red
        }

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor()
        ])

        self.mode = mode
        self.v_size = {
            'digit': 10,
            'image': 3,
            'label': 1
        }
        self.v_type = {
            'digit': sdt.ONE_HOT,
            'image': sdt.IMAGE,
            'label': sdt.BINARY
        }
        self.v_size_high_level = {
            'digit': 10,
            'image': 1,
            'label': 1
        }
        self.v_type_high_level = {
            'digit': sdt.ONE_HOT,
            'image': sdt.BINARY,
            'label': sdt.BINARY
        }
        self.cg = "color_mnist"
        self.cg_high_level = "color_mnist"
        self.cg_projected = "color_mnist_soft"

    def get_sampler_metadata(self, mode):
        o_name = 'image'
        o_type = sdt.IMAGE
        o_size = 3
        if mode == "projected":
            i_type_sampler = {
                'digit': sdt.ONE_HOT,
                'tau_image': sdt.BINARY
            }
            i_size_sampler = {
                'digit': 10,
                'tau_image': 1
            }
            var_mapping = {
                "digit": "digit",
                "image": "tau_image"
            }
        else:
            i_type_sampler = {
                'tau_image': sdt.BINARY
            }
            i_size_sampler = {
                'tau_image': 1
            }
            var_mapping = {
                "image": "tau_image"
            }

        return i_size_sampler, i_type_sampler, o_name, o_size, o_type, var_mapping

    def colorize_image(self, image, color):
        h, w = image.shape
        new_image = np.reshape(image, [h, w, 1])
        new_image = np.concatenate([new_image * color[0], new_image * color[1], new_image * color[2]],
                                   axis=2)
        return new_image

    def sample_digit(self, digit, color):
        total = len(self.raw_mnist_images[digit])
        ind = np.random.randint(0, total)
        img_choice = np.round(self.colorize_image(self.raw_mnist_images[digit][ind], color)).astype(np.uint8)

        return img_choice

    def generate_samples(self, n, U={}, do={}, p_align=0.9, return_U=False, use_tau=False, soft_sampling=False,
                         soft_sampling_mode=None):
        if "u_digit" in U:
            u_digit = U["u_digit"]
        else:
            u_digit = np.random.randint(10, size=n)

        if "u_color_misalign" in U:
            u_color_misalign = U["u_color_misalign"]
        else:
            u_color_misalign = np.random.binomial(1, 1 - p_align, size=n)

        if "u_shade" in U:
            u_shade = U["u_shade"]
        else:
            u_shade = np.random.binomial(1, 0.7, size=n)

        if "u_label_misalign" in U:
            u_label_misalign = U["u_label_misalign"]
        else:
            u_label_misalign = np.random.binomial(1, 1 - p_align, size=n)

        if "digit" in do:
            digit = do["digit"]
        else:
            digit = u_digit

        odd = np.mod(digit, 2)
        odd_flip = np.bitwise_xor(odd, u_color_misalign)
        color = 2 * odd_flip + u_shade

        if "image" in do:
            imgs = []
            for i in range(n):
                bad_case = True
                while bad_case:
                    if (color[i] == 0 or color[i] == 3) and do["image"] == 0:
                        bad_case = False
                    elif (color[i] == 1 or color[i] == 2) and do["image"] == 1:
                        bad_case = False

                    if bad_case:
                        new_u_color_misalign = np.random.binomial(1, 1 - p_align, size=1)[0]
                        new_u_shade = np.random.binomial(1, 0.7, size=1)[0]
                        new_odd_flip = odd[i] | new_u_color_misalign
                        color[i] = 2 * new_odd_flip + new_u_shade

                img_sample = self.sample_digit(digit[i], self.colors[color[i]])
                img_sample = self.transform(img_sample).float()
                if self.normalize:
                    img_sample = 2.0 * img_sample - 1.0
                else:
                    img_sample = 255.0 * img_sample
                imgs.append(img_sample)
        else:
            imgs = []
            for i in range(n):
                img_sample = self.sample_digit(digit[i], self.colors[color[i]])
                img_sample = self.transform(img_sample).float()
                if self.normalize:
                    img_sample = 2.0 * img_sample - 1.0
                else:
                    img_sample = 255.0 * img_sample
                imgs.append(img_sample)

        if "label" in do:
            label = do["label"]
        else:
            color_light_red = (color == 2).astype(int)
            color_dark_red = (color == 3).astype(int)
            color_red = color_light_red + color_dark_red
            label = np.bitwise_xor(color_red, u_label_misalign)
            label = np.expand_dims(label, axis=1)

        one_hot_digit = np.zeros((n, 10))
        one_hot_digit[np.arange(n), digit] = 1

        if self.mode == "sampling_noncausal" or soft_sampling:
            data = {
                'digit': T.tensor(one_hot_digit).float(),
                'image': T.stack(imgs, dim=0)
            }
        else:
            data = {
                'digit': T.tensor(one_hot_digit).float(),
                'image': T.stack(imgs, dim=0),
                'label': T.tensor(label).float()
            }

        if use_tau or soft_sampling:
            color_light_blue = (color == 1).astype(int)
            color_light_red = (color == 2).astype(int)
            shade_light = color_light_blue + color_light_red
            shade_light = np.expand_dims(shade_light, axis=1)

            if soft_sampling:
                data['tau_image'] = T.tensor(shade_light).float()
                if soft_sampling_mode == "basic":
                    del data['digit']
                return data

            data["image"] = T.tensor(shade_light).float()

        if return_U:
            new_U = {
                "u_digit": u_digit,
                "u_color_misalign": u_color_misalign,
                "u_shade": u_shade,
                "u_label_misalign": u_label_misalign
            }
            return data, new_U

        return data

    def calculate_query(self, model=None, tau=False, m=100000, evaluating=False, maximize=False):
        """
        Calculates the query P(Label_{Image = Light} = 1, Digit = 0).
        """
        if model is None:
            raw_samples = self.generate_samples(m, do={"image": 1}, use_tau=tau)
        else:
            if tau:
                do_I = T.ones(m, 1).float()
                raw_samples = model(n=m, do={'image': do_I}, evaluating=evaluating)
            else:
                u_vals = model.pu.sample(m)
                digit_fixed = model(u=u_vals, select={"digit"}, evaluating=True)
                digit_fixed = T.argmax(digit_fixed["digit"], dim=1).cpu().detach().numpy()
                image_do_vals = self.generate_samples(n=m, do={"digit": digit_fixed, "image": 1})["image"]
                raw_samples = model(u=u_vals, do={"image": image_do_vals}, evaluating=evaluating)

        arg_dig = T.argmax(raw_samples["digit"], dim=1)
        cond = (arg_dig == 0)
        new_m = T.sum(cond)
        if new_m == 0:
            if evaluating:
                return T.zeros(1)
            else:
                return None

        samples = {k: v[cond] for (k, v) in raw_samples.items()}

        if evaluating:
            return T.sum(samples['label']) / new_m
        else:
            if maximize:
                loss = T.mean(-T.log(samples['label'] + 1e-8))
            else:
                loss = T.mean(-T.log((1 - samples['label']) + 1e-8))

            return loss

    def get_shade(self, images):
        max_pix = T.amax(images, dim=(1, 2, 3))
        if self.normalize:
            shade = (max_pix > 0.9).float()
        else:
            shade = (max_pix > 240).float()
        return T.unsqueeze(shade, dim=1)

    def sample_ctf(self, q, n=64, batch=None, max_iters=1000, p_align=0.85, normalize=True):
        if batch is None:
            batch = n

        iters = 0
        n_samps = 0
        samples = dict()

        while n_samps < n:
            if iters >= max_iters:
                return float('nan')

            new_samples = self._sample_ctf(batch, q, p_align=p_align, normalize=normalize)
            if isinstance(new_samples, dict):
                if len(samples) == 0:
                    samples = new_samples
                else:
                    for var in new_samples:
                        samples[var] = T.concat((samples[var], new_samples[var]), dim=0)
                        n_samps = len(samples[var])

            iters += 1

        return {var: samples[var][:n] for var in samples}

    def _sample_ctf(self, n, q, p_align=0.85, normalize=True):
        _, U = self.generate_samples(n, return_U=True, p_align=p_align, normalize=normalize)

        n_new = n
        for term in q.cond_term_set:
            samples = self.generate_samples(n=n_new, U=U, do={
                k: expand_do(v, n_new) for (k, v) in term.do_vals.items()
            }, return_U=False, p_align=p_align, normalize=normalize)

            cond_match = T.ones(n_new, dtype=T.bool)
            for (k, v) in term.var_vals.items():
                cond_match *= check_equal(samples[k], v)

            U = {k: v[cond_match] for (k, v) in U.items()}
            n_new = T.sum(cond_match.long()).item()

        if n_new <= 0:
                return float('nan')

        out_samples = dict()
        for term in q.term_set:
            expanded_do_terms = dict()
            for (k, v) in term.do_vals.items():
                    expanded_do_terms[k] = expand_do(v, n_new)
            q_samples = self.generate_samples(n=n_new, U=U, do=expanded_do_terms, return_U=False, p_align=p_align,
                                              normalize=normalize)
            out_samples.update(q_samples)

        return out_samples

    def show_image(self, image, label=None, dir=None):
        if label is not None:
            plt.title('Label is {label}'.format(label=label))
        image = T.movedim(image, 0, -1)
        image = (image + 1.0) / 2.0
        plt.imshow(image)

        if dir is not None:
            plt.savefig(dir)
        else:
            plt.show()
        plt.clf()


if __name__ == "__main__":
    mdg = ColorMNISTDataGenerator(32, "sampling", normalize=True)
    n = 10
    samples, U_vals = mdg.generate_samples(n, use_tau=False, return_U=True)
    for i in range(n):
        print(samples['digit'][i])
        print(samples['label'][i])
        mdg.show_image(samples['image'][i])
    print(mdg.get_shade(samples['image']))

    tau_samples = mdg.generate_samples(n, U=U_vals, use_tau=True)
    print(tau_samples['image'])



    print(mdg.calculate_query(model=None, tau=False, m=100000, evaluating=True))
    print(mdg.calculate_query(model=None, tau=True, m=100000, evaluating=True))
