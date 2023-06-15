import cv2


def highlight_label(start, end, color, freeze_img):
    cv2.rectangle(
        freeze_img,
        start,
        end,
        color,
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


def detected_img_label(coordinate, freeze_img, label):
    cv2.putText(
        freeze_img,
        label,
        coordinate,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
    )


def connected_line(start, end, color, freeze_img):
    cv2.line(
        freeze_img,
        start,
        end,
        color,
        1,
    )


def set_freeze_img(row_start, row_end, col_start, col_end, freeze_img, display_img):
    freeze_img[row_start: row_end, col_start: col_end] = display_img
    return freeze_img


def top_right(x, y, w, h, freeze_img, display_img, pivot_img_size, overlay, opacity, label):
    freeze_img = set_freeze_img(row_start=y - pivot_img_size,
                                row_end=y,
                                col_start=x + w,
                                col_end=x + w + pivot_img_size,
                                freeze_img=freeze_img,
                                display_img=display_img
                                )
    highlight_label(start=(x + w, y),
                    end=(x + w + pivot_img_size, y + 20),
                    color=(46, 200, 255),
                    freeze_img=freeze_img)
    detected_img(freeze_img=freeze_img, overlay=overlay, opacity=opacity)
    detected_img_label(coordinate=(x + w, y + 10), freeze_img=freeze_img, label=label)
    connected_line(start=(x + int(w / 2), y),
                   end=(x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                   color=(67, 67, 67),
                   freeze_img=freeze_img)
    connected_line(start=(x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                   end=(x + w, y - int(pivot_img_size / 2)),
                   color=(67, 67, 67),
                   freeze_img=freeze_img)


def bottom_left(x, y, w, h, freeze_img, display_img, pivot_img_size, overlay, opacity, label):
    freeze_img = set_freeze_img(row_start=y + h,
                                row_end=y + h + pivot_img_size,
                                col_start=x - pivot_img_size,
                                col_end=x,
                                freeze_img=freeze_img,
                                display_img=display_img)
    highlight_label(start=(x - pivot_img_size, y + h - 20),
                    end=(x, y + h),
                    color=(46, 200, 255),
                    freeze_img=freeze_img)
    detected_img(freeze_img=freeze_img, overlay=overlay, opacity=opacity)
    detected_img_label(coordinate=(x - pivot_img_size, y + h - 10), freeze_img=freeze_img, label=label)
    connected_line(start=(x + int(w / 2), y + h),
                   end=(x + int(w / 2) - int(w / 4), y + h + int(pivot_img_size / 2)),
                   color=(67, 67, 67),
                   freeze_img=freeze_img)
    connected_line(start=(x + int(w / 2) - int(w / 4), y + h + int(pivot_img_size / 2)),
                   end=(x, y + h + int(pivot_img_size / 2)),
                   color=(67, 67, 67),
                   freeze_img=freeze_img)


def top_left(x, y, w, h, freeze_img, display_img, pivot_img_size, overlay, opacity, label):
    freeze_img = set_freeze_img(row_start=y - pivot_img_size,
                                row_end=y,
                                col_start=x - pivot_img_size,
                                col_end=x,
                                freeze_img=freeze_img,
                                display_img=display_img)
    highlight_label(start=(x - pivot_img_size, y),
                    end=(x, y + 20),
                    color=(46, 200, 255),
                    freeze_img=freeze_img)
    detected_img(freeze_img=freeze_img, overlay=overlay, opacity=opacity)
    detected_img_label(coordinate=(x - pivot_img_size, y + 10), freeze_img=freeze_img, label=label)
    connected_line(start=(x + int(w / 2), y),
                   end=(x + int(w / 2) - int(w / 4), y - int(pivot_img_size / 2)),
                   color=(67, 67, 67),
                   freeze_img=freeze_img)
    connected_line(start=(x + int(w / 2) - int(w / 4), y - int(pivot_img_size / 2)),
                   end=(x, y - int(pivot_img_size / 2)),
                   color=(67, 67, 67),
                   freeze_img=freeze_img)


def bottom_right(x, y, w, h, freeze_img, display_img, pivot_img_size, overlay, opacity, label):
    freeze_img = set_freeze_img(row_start=y + h,
                                row_end=y + h + pivot_img_size,
                                col_start=x + w,
                                col_end=x + w + pivot_img_size,
                                freeze_img=freeze_img,
                                display_img=display_img)
    highlight_label(start=(x + w, y + h - 20),
                    end=(x + w + pivot_img_size, y + h),
                    color=(46, 200, 255),
                    freeze_img=freeze_img)
    detected_img(freeze_img=freeze_img, overlay=overlay, opacity=opacity)
    detected_img_label(coordinate=(x + w, y + h - 10), freeze_img=freeze_img, label=label)
    connected_line(start=(x + int(w / 2), y + h),
                   end=(x + int(w / 2) + int(w / 4), y + h + int(pivot_img_size / 2)),
                   color=(67, 67, 67),
                   freeze_img=freeze_img)
    connected_line(start=(x + int(w / 2) + int(w / 4), y + h + int(pivot_img_size / 2)),
                   end=(x + w, y + h + int(pivot_img_size / 2)),
                   color=(67, 67, 67),
                   freeze_img=freeze_img)
    connected_line(start=(x + int(w / 2), y),
                   end=(x + int(w / 2) - int(w / 4), y - int(pivot_img_size / 2)),
                   color=(67, 67, 67),
                   freeze_img=freeze_img)


def img_and_label(x, y, w, h, freeze_img, display_img, pivot_img_size, label, resolution_x, resolution_y):
    overlay = freeze_img.copy()
    opacity = 0.4

    if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x: # top right
        top_right(x, y, w, h, freeze_img, display_img, pivot_img_size, overlay, opacity, label)
    elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0: # bottom left
        bottom_left(x, y, w, h, freeze_img, display_img, pivot_img_size, overlay, opacity, label)
    elif y - pivot_img_size > 0 and x - pivot_img_size > 0: # top left
        top_left(x, y, w, h, freeze_img, display_img, pivot_img_size, overlay, opacity, label)
    elif x + w + pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y: # bottom right
        bottom_right(x, y, w, h, freeze_img, display_img, pivot_img_size, overlay, opacity, label)

    return freeze_img, label

