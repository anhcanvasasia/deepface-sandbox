def highlight_label(x, y, w, h, freeze_img, pivot_img_size):
    cv2.rectangle(
        freeze_img,
        (x + w, y + h - 20),
        (x + w + pivot_img_size, y + h),
        (46, 200, 255),
        cv2.FILLED,
    )


def detected_img(freeze_img, overlay, opacity):
    cv2.addWeighted(
        overlay,
        opacity,
        freeze_img,
        1 - opacity,
        0,
        freeze_img,
    )


def detected_img_label(x, y, w, h, freeze_img):
    cv2.putText(
        freeze_img,
        label,
        (x + w, y + h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
    )


def connected_line_diagonal(x, y, w, h, freeze_img, pivot_img_size):
    cv2.line(
        freeze_img,
        (x + int(w / 2), y + h),
        (
            x + int(w / 2) + int(w / 4),
            y + h + int(pivot_img_size / 2),
        ),
        (67, 67, 67),
        1,
    )


def connected_line_dash(x, y, w, h, freeze_img, pivot_img_size):
    cv2.line(
        freeze_img,
        (
            x + int(w / 2) + int(w / 4),
            y + h + int(pivot_img_size / 2),
        ),
        (x + w, y + h + int(pivot_img_size / 2)),
        (67, 67, 67),
        1,
    )


def img_and_label(x, y, w, h, freeze_img, pivot_img_size):
    # bottom righ
    freeze_img[
    y + h: y + h + pivot_img_size,
    x + w: x + w + pivot_img_size,
    ] = display_img

    overlay = freeze_img.copy()
    opacity = 0.4

    highlight_label(x=x, y=y, w=w, h=h, freeze_img=freeze_img, pivot_img_size=pivot_img_size)
    detected_img(freeze_img=freeze_img, overlay=overlay, opacity=opacity)
    detected_img_label(x=x, y=y, w=w, h=h, freeze_img=freeze_img)
    connected_line_diagonal(x=x, y=y, w=w, h=h, freeze_img=freeze_img, pivot_img_size=pivot_img_size)
    connected_line_dash(x=x, y=y, w=w, h=h, freeze_img=freeze_img, pivot_img_size=pivot_img_size)