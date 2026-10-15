"""Export a Markdown copy with GitHub raw image URLs; keep source untouched."""
import argparse
from pathlib import Path
import re
from urllib.parse import quote,urlsplit

ROOT=Path(__file__).resolve().parent.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',default='main',help='Git branch or commit SHA for images')
    parser.add_argument('--output',type=Path,default=ROOT/'exports/新实验报告_课程网站版.md')
    args=parser.parse_args()
    source=ROOT/'新实验报告.md'
    if args.output.resolve()==source.resolve():
        parser.error('Choose a separate output file to preserve relative links in the source.')
    text=source.read_text(encoding='utf-8')
    def convert(match):
        alt,path=match.groups()
        if urlsplit(path).scheme:return match.group(0)
        image=(ROOT/path).resolve()
        if not image.is_relative_to(ROOT) or not image.is_file():
            raise ValueError('Invalid or missing image: '+path)
        url='https://raw.githubusercontent.com/70clementine/Bert-Lab/'+quote(args.ref,safe='')+'/'+quote(image.relative_to(ROOT).as_posix(),safe='/')
        return f'![{alt}]({url})'
    text=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',convert,text)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text,encoding='utf-8')
    print('Exported:',args.output)


if __name__=='__main__':main()
