#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <time.h>

/* Note: This code is meant for Linux bare-metal systems with PHC-capable NICs. */
#ifdef __linux__
#include <linux/net_tstamp.h>
#include <linux/sockios.h>
#else
/* Dummy definitions for macOS/non-Linux to allow viewing the boilerplate structure.
 * This will not function as an actual hardware timestamping receiver outside of Linux. */
#define SOF_TIMESTAMPING_RX_HARDWARE 0
#define SOF_TIMESTAMPING_RAW_HARDWARE 0
#define SOF_TIMESTAMPING_SYS_HARDWARE 0
#define SOF_TIMESTAMPING_SOFTWARE 0
#define SO_TIMESTAMPING 37
#define SCM_TIMESTAMPING SO_TIMESTAMPING
#endif

#define PORT 9000
#define INTERFACE "eth0"
#define BUF_SIZE 2048

/*
 * Detailed explanation of PHC vs System Clock:
 * 
 * The System Clock (CLOCK_REALTIME) is maintained by the OS kernel. It is subject to 
 * NTP adjustments, leap seconds, and OS scheduler jitter. Standard software timestamps 
 * (like time.time() in Python or gettimeofday in C) read this clock.
 * 
 * The PTP Hardware Clock (PHC) resides directly on the Network Interface Card (NIC). 
 * It runs independently of the host CPU and OS. When an Ethernet frame arrives at the 
 * physical layer (PHY) or MAC, the hardware immediately captures the exact moment using 
 * the PHC. This timestamp is stored in a register and later passed up the stack via 
 * socket ancillary data (control messages). 
 * 
 * By using the PHC, we bypass the OS entirely for timing. We eliminate all software-induced 
 * jitter and variable latency, guaranteeing deterministic timing critical for causality 
 * in high-frequency, continuous biological data streams.
 */

int main() {
    int sockfd;
    struct sockaddr_in servaddr;
    struct ifreq ifr;

    // 1. Create a UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // 2. Bind to a specific interface (e.g., eth0)
    memset(&ifr, 0, sizeof(ifr));
    snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), INTERFACE);
    
#ifdef SO_BINDTODEVICE
    // Bind the socket strictly to the designated physical interface
    if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, (void *)&ifr, sizeof(ifr)) < 0) {
        perror("Warning: Bind to device failed (requires root privileges usually)");
    }
#endif

    // Bind to the designated port
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(PORT);

    if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
        perror("Bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    // 3. Enable SO_TIMESTAMPING socket options.
    /*
     * Memory Structures and Flags:
     * SOF_TIMESTAMPING_RX_HARDWARE: Request hardware timestamps for incoming packets.
     * SOF_TIMESTAMPING_RAW_HARDWARE: Report the raw hardware timestamp rather than a timestamp 
     *                                converted to the system time base. This gives us the pure PHC value.
     * SOF_TIMESTAMPING_SOFTWARE: Fallback/comparison to get standard kernel software timestamps.
     */
    int flags = SOF_TIMESTAMPING_RX_HARDWARE | SOF_TIMESTAMPING_RAW_HARDWARE | 
                SOF_TIMESTAMPING_SYS_HARDWARE | SOF_TIMESTAMPING_SOFTWARE;
                
    if (setsockopt(sockfd, SOL_SOCKET, SO_TIMESTAMPING, &flags, sizeof(flags)) < 0) {
        perror("Warning: setsockopt SO_TIMESTAMPING failed");
        // Do not exit, allow demonstration to continue even if NIC doesn't support it
    }

    printf("Listening for UDP packets on %s port %d with HW Timestamping requested...\n", INTERFACE, PORT);

    char buffer[BUF_SIZE];
    char ctrl[2048]; // Buffer for ancillary data (control messages)
    struct iovec iov;
    struct msghdr msg;
    struct sockaddr_in clientaddr;

    iov.iov_base = buffer;
    iov.iov_len = BUF_SIZE;

    memset(&msg, 0, sizeof(msg));
    msg.msg_name = &clientaddr;
    msg.msg_namelen = sizeof(clientaddr);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    // 4. Implement the recvmsg loop
    while (1) {
        msg.msg_control = ctrl;
        msg.msg_controllen = sizeof(ctrl); // Reset for each receive
        
        ssize_t n = recvmsg(sockfd, &msg, 0);
        if (n < 0) {
            perror("recvmsg failed");
            continue;
        }

        // 5. Extract the hardware timestamp from ancillary data
        struct cmsghdr *cmsg;
        struct timespec *hw_ts = NULL;

        /*
         * We iterate through the control messages (cmsghdr) populated by the kernel.
         * A single recvmsg call can return multiple control messages in its ancillary data buffer.
         * We specifically look for the SOL_SOCKET level and SCM_TIMESTAMPING (often synonymous with SO_TIMESTAMPING) type.
         */
        for (cmsg = CMSG_FIRSTHDR(&msg); cmsg != NULL; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_TIMESTAMPING) {
                /*
                 * The data payload of SCM_TIMESTAMPING contains an array of 3 struct timespecs.
                 * struct timespec ts[3]:
                 * ts[0] = Software timestamp (if SOF_TIMESTAMPING_SOFTWARE was set)
                 * ts[1] = Hardware timestamp converted to system clock (deprecated, unreliable)
                 * ts[2] = Raw Hardware timestamp (if SOF_TIMESTAMPING_RAW_HARDWARE was set)
                 * 
                 * We extract ts[2] to retrieve our pure, deterministic PHC timestamp.
                 */
                struct timespec *ts = (struct timespec *)CMSG_DATA(cmsg);
                hw_ts = &ts[2];
                break;
            }
        }

        // Print payload size and the extracted timestamp
        printf("Received packet: Payload Size = %zd bytes\n", n);
        if (hw_ts != NULL && (hw_ts->tv_sec != 0 || hw_ts->tv_nsec != 0)) {
            printf(" -> Raw HW Timestamp (PHC): %ld seconds, %ld nanoseconds\n", (long)hw_ts->tv_sec, (long)hw_ts->tv_nsec);
        } else {
            printf(" -> HW Timestamp: Not available (Packet captured using software clock)\n");
        }
        printf("--------------------------------------------------\n");
    }

    close(sockfd);
    return 0;
}
