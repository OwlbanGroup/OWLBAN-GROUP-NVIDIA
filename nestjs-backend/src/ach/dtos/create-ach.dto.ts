import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { IsEnum, IsString, IsNumber, IsBoolean, IsOptional, IsDateString, Min, Max, Matches } from 'class-validator';
import { AchSecCode, AchTransactionType } from '../entities/ach-payment.entity';

export class CreateAchDto {
  @ApiProperty({
    description: 'SEC code for ACH transaction',
    enum: AchSecCode,
    example: AchSecCode.PPD,
  })
  @IsEnum(AchSecCode)
  secCode: AchSecCode;

  @ApiProperty({
    description: 'Transaction type (credit or debit)',
    enum: AchTransactionType,
    example: AchTransactionType.CREDIT,
  })
  @IsEnum(AchTransactionType)
  transactionType: AchTransactionType;

  @ApiProperty({
    description: 'Amount in cents',
    example: 100000,
    minimum: 1,
    maximum: 100000000,
  })
  @IsNumber()
  @Min(1)
  @Max(100000000)
  amountCents: number;

  @ApiProperty({
    description: 'Originator name',
    example: 'ACME Corporation',
    maxLength: 16,
  })
  @IsString()
  originatorName: string;

  @ApiProperty({
    description: 'Originator ID (10 digits)',
    example: '1234567890',
  })
  @IsString()
  @Matches(/^\d{10}$/, { message: 'Originator ID must be 10 digits' })
  originatorId: string;

  @ApiProperty({
    description: 'Receiver name',
    example: 'John Doe',
    maxLength: 22,
  })
  @IsString()
  receiverName: string;

  @ApiProperty({
    description: 'Receiver account number',
    example: '123456789',
    maxLength: 17,
  })
  @IsString()
  receiverAccountNumber: string;

  @ApiProperty({
    description: 'Receiver routing number (9 digits)',
    example: '021000021',
  })
  @IsString()
  @Matches(/^\d{9}$/, { message: 'Routing number must be 9 digits' })
  receiverRoutingNumber: string;

  @ApiPropertyOptional({
    description: 'Addenda record (additional information)',
    example: 'Invoice #12345',
    maxLength: 80,
  })
  @IsOptional()
  @IsString()
  addendaRecord?: string;

  @ApiPropertyOptional({
    description: 'Same-day ACH processing',
    example: false,
  })
  @IsOptional()
  @IsBoolean()
  sameDayAch?: boolean;

  @ApiPropertyOptional({
    description: 'Effective date (YYYY-MM-DD)',
    example: '2026-01-15',
  })
  @IsOptional()
  @IsDateString()
  effectiveDate?: string;

  @ApiPropertyOptional({
    description: 'External reference',
    example: 'INV-2024-001',
  })
  @IsOptional()
  @IsString()
  externalReference?: string;

  @ApiPropertyOptional({
    description: 'Idempotency key for duplicate prevention',
    example: 'ach-20260102-001',
  })
  @IsOptional()
  @IsString()
  idempotencyKey?: string;

  @ApiPropertyOptional({
    description: 'Organization ID',
    example: 'org-123',
  })
  @IsOptional()
  @IsString()
  organizationId?: string;
}
