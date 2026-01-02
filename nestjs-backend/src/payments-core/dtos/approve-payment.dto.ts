import {
  IsNotEmpty,
  IsOptional,
  IsString,
  IsUUID,
  MaxLength,
} from 'class-validator';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

export class ApprovePaymentDto {
  @ApiProperty({
    description: 'Payment ID to approve (UUID)',
    example: '123e4567-e89b-12d3-a456-426614174000',
  })
  @IsUUID()
  @IsNotEmpty()
  paymentId: string;

  @ApiPropertyOptional({
    description: 'Comments from the approver',
    example: 'Approved for payment processing',
  })
  @IsString()
  @IsOptional()
  comments?: string;

  @ApiPropertyOptional({
    description: 'IP address of the approver (auto-captured)',
    example: '192.168.1.1',
  })
  @IsString()
  @IsOptional()
  @MaxLength(45)
  ipAddress?: string;

  @ApiPropertyOptional({
    description: 'User agent of the approver (auto-captured)',
    example: 'Mozilla/5.0...',
  })
  @IsString()
  @IsOptional()
  userAgent?: string;
}

export class RejectPaymentDto {
  @ApiProperty({
    description: 'Payment ID to reject (UUID)',
    example: '123e4567-e89b-12d3-a456-426614174000',
  })
  @IsUUID()
  @IsNotEmpty()
  paymentId: string;

  @ApiProperty({
    description: 'Reason for rejection',
    example: 'Insufficient documentation provided',
  })
  @IsString()
  @IsNotEmpty()
  rejectionReason: string;

  @ApiPropertyOptional({
    description: 'Additional comments from the approver',
    example: 'Please provide invoice copy',
  })
  @IsString()
  @IsOptional()
  comments?: string;

  @ApiPropertyOptional({
    description: 'IP address of the approver (auto-captured)',
    example: '192.168.1.1',
  })
  @IsString()
  @IsOptional()
  @MaxLength(45)
  ipAddress?: string;

  @ApiPropertyOptional({
    description: 'User agent of the approver (auto-captured)',
    example: 'Mozilla/5.0...',
  })
  @IsString()
  @IsOptional()
  userAgent?: string;
}
