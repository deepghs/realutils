from plot import image_plot

if __name__ == '__main__':
    image_plot(
        *['idolsankaku/1.jpg', 'idolsankaku/7.jpg'],
        columns=2,
        figsize=(4, 8),
    )
