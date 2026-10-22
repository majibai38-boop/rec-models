"""Encode an explicitly labelled slideshow from unaltered Web screenshots."""
from pathlib import Path
from PIL import Image


def main():
    root = Path(__file__).resolve().parent
    names = ['01-upload', '02-scope', '03-summary', '04-types', '05-details']
    frames = []
    for name in names:
        with Image.open(root / (name + '.jpg')) as image:
            frames.append(image.convert('RGB'))
    size = (max(f.width for f in frames), max(f.height for f in frames))
    canvases = []
    for frame in frames:
        canvas = Image.new('RGB', size, 'white')
        canvas.paste(frame, (0, 0))
        canvases.append(canvas.quantize(colors=192))
    target = root / 'workbench-demo.gif'
    canvases[0].save(target, save_all=True, append_images=canvases[1:], duration=3000,
                     loop=0, optimize=True, disposal=2)
    with Image.open(target) as result:
        assert result.n_frames == len(names)
    print(f'{target.name}: {target.stat().st_size:,} bytes, {len(names)} frames')


if __name__ == '__main__':
    main()
