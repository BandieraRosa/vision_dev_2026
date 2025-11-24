#ifndef RM_SERIAL_DRIVER__CRC_HPP_
#define RM_SERIAL_DRIVER__CRC_HPP_

#include <cstdint>

namespace crc16
{
/**
 * @brief CRC16 Verify function
 * @param[in] pchMessage : Data to Verify,
 * @param[in] dwLength : Stream length = Data + checksum
 * @return : True or False (CRC Verify Result)
 */
uint32_t verify_crc16_check_sum(const uint8_t* pchMessage, uint32_t dwLength);  // 验证
inline uint16_t crc16_byte(uint16_t crc, const uint8_t DATA);                   // 计算crc
uint16_t crc16_calc(const uint8_t* buf, std::size_t len, uint16_t crc);         // 计算crc
bool crc16_verify(const uint8_t* buf, std::size_t len);  // pan duan
/**
 * @brief Append CRC16 value to the end of the buffer
 * @param[in] pchMessage : Data to Verify,要验证的数据
 * @param[in] dwLength : Stream length = Data + checksum
 * @return none
 */
void append_crc16_check_sum(uint8_t* pchMessage, uint32_t dwLength);  // 追加

}  // namespace crc16

#endif  // RM_SERIAL_DRIVER__CRC_HPP_
