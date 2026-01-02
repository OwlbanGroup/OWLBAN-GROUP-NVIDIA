import {
  IsEnum,
  IsNotEmpty,
  IsNumber,
  IsOptional,
  IsString,
  IsUUID,
  Min,
  MaxLength,
  IsObject,
} from 'class-validator';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { PaymentType } from '../enums/payment-type.enum';
import { PaymentDirection } from '../enums/payment-direction.enum';

export class CreatePaymentDto {
  @ApiProperty({
    description: 'External reference for the payment (your system reference)',
    example: 'INV-2024-001',
  })
  @IsString()
  @IsNotEmpty()
  @MaxLength(255)
  externalRef: string;

  @ApiProperty({
    description: 'Payment type (ACH, WIRE, RTP)',
    enum: PaymentType,
    example: PaymentType.ACH,
  })
  @IsEnum(PaymentType)
  @IsNotEmpty()
  type: PaymentType;

  @ApiProperty({
    description: 'Payment direction (CREDIT = outgoing, DEBIT = incoming)',
    enum: PaymentDirection,
    example: PaymentDirection.CREDIT,
  })
  @IsEnum(PaymentDirection)
  @IsNotEmpty()
  direction: PaymentDirection;

  @ApiProperty({
    description: 'Payment amount in cents (to avoid floating point issues)',
    example: 100000,
    minimum: 1,
  })
  @IsNumber()
  @Min(1)
  @IsNotEmpty()
  amountCents: number;

  @ApiPropertyOptional({
    description: 'Currency code (ISO 4217)',
    example: 'USD',
    default: 'USD',
  })
  @IsString()
  @IsOptional()
  @MaxLength(3)
  currency?: string;

  @ApiPropertyOptional({
    description: 'Source account ID (UUID)',
    example: '123e4567-e89b-12d3-a456-426614174000',
  })
  @IsUUID()
  @IsOptional()
  fromAccountId?: string;

  @ApiProperty({
    description: 'Destination account number',
    example: '1234567890',
  })
  @IsString()
  @IsNotEmpty()
  @MaxLength(255)
  toAccountNumber: string;

  @ApiProperty({
    description: 'Destination routing number',
    example: '021000021',
  })
  @IsString()
  @IsNotEmpty()
  @MaxLength(255)
  toRoutingNumber: string;

  @ApiProperty({
    description: 'Destination account name',
    example: 'John Doe',
  })
  @IsString()
  @IsNotEmpty()
  @MaxLength(255)
  toAccountName: string;

  @ApiPropertyOptional({
    description: 'Destination bank name',
    example: 'JPMorgan Chase Bank',
  })
  @IsString()
  @IsOptional()
  @MaxLength(255)
  toBankName?: string;

  @ApiPropertyOptional({
    description: 'Payment description/memo',
    example: 'Payment for Invoice INV-2024-001',
  })
  @IsString()
  @IsOptional()
  description?: string;

  @ApiPropertyOptional({
    description: 'Reference/invoice number',
    example: 'INV-2024-001',
  })
  @IsString()
  @IsOptional()
  @MaxLength(255)
  referenceNumber?: string;

  @ApiProperty({
    description: 'Organization ID (UUID)',
    example: '123e4567-e89b-12d3-a456-426614174000',
  })
  @IsUUID()
  @IsNotEmpty()
  organizationId: string;

  @ApiPropertyOptional({
    description: 'Additional metadata (flexible JSON object)',
    example: { invoiceId: '12345', customerId: '67890' },
  })
  @IsObject()
  @IsOptional()
  metadata?: any;

  // Helper method to get amount in dollars
  getAmountDollars(): number {
    return this.amountCents / 100;
  }

  // Helper method to validate amount
  isValidAmount(): boolean {
    return this.amountCents > 0 && Number.isInteger(this.amountCents);
  }
}
