
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../lib/mcraw.h"

// Read and decode a mcraw file
// output metadata

//-----------------------------------------------------------------------------
void print_metadata(mr_ctx_t *ctx)
{
    printf("   Video frames: %d\n", mr_get_frame_count(ctx));
    printf("  Audio packets: %d\n", mr_get_audio_packet_count(ctx));

    printf("          Width: %d\n", mr_get_width(ctx));
    printf("         Height: %d\n", mr_get_height(ctx));
    printf(" Bits per pixel: %d\n", mr_get_bits_per_pixel(ctx));
    printf("  Stored format: raw%d\n", mr_get_stored_format(ctx));

    uint32_t cfa_pattern = mr_get_cfa_pattern(ctx);

    if (cfa_pattern == 0x01000201) {
        printf("    CFA pattern: gbrg\n");
    }
    else if (cfa_pattern == 0x00010102) {
        printf("    CFA pattern: bggr\n");
    }
    else if (cfa_pattern == 0x01020001) {
        printf("    CFA pattern: grbg\n");
    }
    else {  // 0x02010100
        printf("    CFA pattern: rggb\n");
    }

    printf("    Black level: %d\n", mr_get_black_level(ctx));
    printf("    White level: %d\n", mr_get_white_level(ctx));

    int num = 0, den = 1;
    mr_get_frame_rate(ctx, &num, &den);
    printf("   Frame rate:   %d/%d\n", num, den);

    printf("   Focal length: %f\n", mr_get_focal_length(ctx));
    printf("       Aperture: %f\n", mr_get_aperture(ctx));
    printf("            Iso: %d\n", mr_get_iso(ctx));

    int64_t ns = mr_get_exposure_time(ctx);
    printf("       Exposure: 1/%ld (%ld)\n", 1000000000 / ns, ns);

    double *mat;

    mat = mr_get_color_matrix1(ctx);
    printf("  Color matrix1: %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
           mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]);

    mat = mr_get_color_matrix2(ctx);
    printf("  Color matrix2: %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
           mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]);

    mat = mr_get_forward_matrix1(ctx);
    printf("Forward matrix1: %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
           mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]);

    mat = mr_get_forward_matrix2(ctx);
    printf("Forward matrix2: %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
           mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]);

    time_t t = mr_get_timestamp(ctx) / 1000;
    struct tm *tm_info = localtime(&t);

    char buffer[26];
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    printf("      Timestamp: %s\n", buffer);
    printf("   Manufacturer: %s\n", mr_get_manufacturer(ctx));
    printf("   Manufacturer: %s\n", mr_get_model(ctx));

    printf("\n");
}

//-----------------------------------------------------------------------------
void cmp_meta_data(uint32_t frame_idx, mr_frame_data_t *a, mr_frame_data_t *b)
{
    a->timestamp = b->timestamp = 0;

    int cmp = memcmp(a, b, sizeof(mr_frame_data_t));

    // Metadata blocks are identical
    if (cmp == 0 || frame_idx == 0) {
        return;
    }

    printf("Frame %d metadata changes\n", frame_idx);

    if (a->width != b->width || a->height != b->height) {
        printf("       Resolution: %d x %d  ->  %d x %d\n", a->width, a->height, b->width, b->height);
    }

    if (a->original_width != b->original_width || a->original_height != b->original_height) {
        printf("   Org Resolution: %d x %d  ->  %d x %d\n", a->original_width, a->original_height, b->original_width, b->original_height);
    }

    if (a->stored_pixel_format != b->stored_pixel_format) {
        printf("    Stored format: %d  ->  %d\n", a->stored_pixel_format, b->stored_pixel_format);
    }

    if (a->iso != b->iso) {
        printf("              Iso: %d  ->  %d\n", a->iso, b->iso);
    }

    if (a->exposure_time != b->exposure_time)
    {
        printf("         Exposure: 1/%ld (%ld)  ->  1/%ld (%ld)\n",
               1000000000 / a->exposure_time, a->exposure_time,
               1000000000 / b->exposure_time, b->exposure_time);
    }

    if (a->orientation != b->orientation) {
        printf("      Orientation: %d  ->  %d\n", a->orientation, b->orientation);
    }

    if (a->as_shot_neutral[0] != b->as_shot_neutral[0] || a->as_shot_neutral[1] != b->as_shot_neutral[1] || a->as_shot_neutral[2] != b->as_shot_neutral[2]) {
        printf("  As shot neutral: %f, %f, %f  ->  %f, %f, %f\n",
               a->as_shot_neutral[0], a->as_shot_neutral[1], a->as_shot_neutral[2],
               b->as_shot_neutral[0], b->as_shot_neutral[1], b->as_shot_neutral[2]);
    }

    printf("\n");
}

//-----------------------------------------------------------------------------
int main(int argc, const char * argv[])
{
    mr_ctx_t *ctx;

    if (argc < 2)
    {
        printf("Not enough parameters!\n Usage: %s filename.mcraw <1|0 dump file structure>", argv[0]);
        exit(0);
    }

    const char *mcraw_filename  = argv[1];
    int dump_file = argc > 2 ? (argv[2][0] - '0') : 0;
    ctx = mr_decoder_new(0);

    if (ctx == NULL) {
        return -1;
    }

    int res = mr_decoder_open(ctx, mcraw_filename);
    if (res != 0) {
        goto done;
    }

    res = mr_decoder_parse(ctx);
    if (res != 0) {
        goto done;
    }

    print_metadata(ctx);

    if (dump_file) {
        mr_dump(ctx);
    }

    {
        FILE *fd = mr_get_file_handle(ctx);

        uint32_t nb_frames = mr_get_frame_count(ctx);
        mr_buffer_offset_t *video_index = mr_get_index(ctx);

        int width  = mr_get_width(ctx);
        int height = mr_get_height(ctx);

        mr_packet_t pkt_frame = {};
        mr_frame_data_t prev_meta_data = {};
        mr_frame_data_t meta_data = {};

        // Raw bayer image
        size_t raw_size = width * height * sizeof(uint16_t);
        uint8_t *raw_data = malloc(raw_size);

        size_t bayer_frame_size = 0;

        for (uint32_t i = 0; i < nb_frames; i++)
        {
            if (mr_read_video_frame(fd, video_index[i].offset, &pkt_frame) == 0)
            {
                res = mr_read_frame_metadata(fd, &meta_data);
                if (res != 0) {
                    break;
                }

                cmp_meta_data(i, &prev_meta_data, &meta_data);
                prev_meta_data = meta_data;

                bayer_frame_size = mr_decode_video_frame(raw_data, pkt_frame.data, pkt_frame.size, width, height);
                if (bayer_frame_size == 0) {
                    break;
                }
            }
        }

        mr_packet_free(&pkt_frame);

        free(raw_data);
    }

done:
    mr_decoder_close(ctx);
    mr_decoder_free(ctx);

    return 0;
}
