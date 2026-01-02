import { Injectable, BadRequestException } from '@nestjs/common';
import { CreateAchDto } from '../dtos/create-ach.dto';

@Injectable()
export class AchValidationService {
  /**
   * Validate routing number using ABA checksum algorithm
   */
  validateRoutingNumber(routingNumber: string): boolean {
    if (!/^\d{9}$/.test(routingNumber)) {
      throw new BadRequestException('Routing number must be 9 digits');
    }

    // ABA checksum validation
    const digits = routingNumber.split('').map(Number);
    const checksum = 
      (3 * (digits[0] + digits[3] + digits[6])) +
      (7 * (digits[1] + digits[4] + digits[7])) +
      (1 * (digits[2] + digits[5] + digits[8]));

    if (checksum % 10 !== 0) {
      throw new BadRequestException('Invalid routing number checksum');
    }

    return true;
  }

  /**
   * Validate account number format
   */
  validateAccountNumber(accountNumber: string): boolean {
    if (!/^\d{1,17}$/.test(accountNumber)) {
      throw new BadRequestException('Account number must be 1-17 digits');
    }
    return true;
  }

  /**
   * Validate ACH amount limits
   */
  validateAmount(amountCents: number, sameDayAch: boolean): boolean {
    const maxAmount = sameDayAch ? 100000000 : 1000000000; // $1M for same-day, $10M for standard
    
    if (amountCents < 1) {
      throw new BadRequestException('Amount must be at least $0.01');
    }

    if (amountCents > maxAmount) {
      const maxDollars = maxAmount / 100;
      throw new BadRequestException(`Amount exceeds maximum of $${maxDollars.toLocaleString()}`);
    }

    return true;
  }

  /**
   * Validate effective date
   */
  validateEffectiveDate(effectiveDate: string, sameDayAch: boolean): boolean {
    const effectiveDateObj = new Date(effectiveDate);
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    if (effectiveDateObj < today) {
      throw new BadRequestException('Effective date cannot be in the past');
    }

    // Max 2 business days in future for same-day ACH
    if (sameDayAch) {
      const maxDate = new Date(today);
      maxDate.setDate(maxDate.getDate() + 2);
      
      if (effectiveDateObj > maxDate) {
        throw new BadRequestException('Same-day ACH effective date cannot be more than 2 days in future');
      }
    }

    // Max 30 days in future for standard ACH
    const maxFutureDate = new Date(today);
    maxFutureDate.setDate(maxFutureDate.getDate() + 30);
    
    if (effectiveDateObj > maxFutureDate) {
      throw new BadRequestException('Effective date cannot be more than 30 days in future');
    }

    return true;
  }

  /**
   * Validate originator ID format
   */
  validateOriginatorId(originatorId: string): boolean {
    if (!/^\d{10}$/.test(originatorId)) {
      throw new BadRequestException('Originator ID must be 10 digits');
    }
    return true;
  }

  /**
   * Validate SEC code
   */
  validateSecCode(secCode: string): boolean {
    const validSecCodes = ['PPD', 'CCD', 'WEB', 'TEL', 'CTX'];
    
    if (!validSecCodes.includes(secCode)) {
      throw new BadRequestException(`Invalid SEC code. Must be one of: ${validSecCodes.join(', ')}`);
    }

    return true;
  }

  /**
   * Validate transaction type
   */
  validateTransactionType(transactionType: string): boolean {
    const validTypes = ['CREDIT', 'DEBIT'];
    
    if (!validTypes.includes(transactionType)) {
      throw new BadRequestException(`Invalid transaction type. Must be one of: ${validTypes.join(', ')}`);
    }

    return true;
  }

  /**
   * Validate addenda record length
   */
  validateAddenda(addendaRecord?: string): boolean {
    if (addendaRecord && addendaRecord.length > 80) {
      throw new BadRequestException('Addenda record cannot exceed 80 characters');
    }
    return true;
  }

  /**
   * Validate name fields
   */
  validateName(name: string, fieldName: string): boolean {
    if (!name || name.trim().length === 0) {
      throw new BadRequestException(`${fieldName} is required`);
    }

    if (name.length > 22) {
      throw new BadRequestException(`${fieldName} cannot exceed 22 characters`);
    }

    // Check for invalid characters (only alphanumeric and spaces allowed)
    if (!/^[a-zA-Z0-9\s]+$/.test(name)) {
      throw new BadRequestException(`${fieldName} can only contain letters, numbers, and spaces`);
    }

    return true;
  }

  /**
   * Validate complete ACH payment
   */
  async validateAchPayment(dto: CreateAchDto): Promise<void> {
    // Validate SEC code
    this.validateSecCode(dto.secCode);

    // Validate transaction type
    this.validateTransactionType(dto.transactionType);

    // Validate routing number
    this.validateRoutingNumber(dto.receiverRoutingNumber);

    // Validate account number
    this.validateAccountNumber(dto.receiverAccountNumber);

    // Validate amount
    this.validateAmount(dto.amountCents, dto.sameDayAch || false);

    // Validate effective date (if provided)
    if (dto.effectiveDate) {
      this.validateEffectiveDate(dto.effectiveDate, dto.sameDayAch || false);
    }

    // Validate originator ID
    this.validateOriginatorId(dto.originatorId);

    // Validate originator name
    this.validateName(dto.originatorName, 'Originator name');

    // Validate receiver name
    this.validateName(dto.receiverName, 'Receiver name');

    // Validate addenda (if provided)
    if (dto.addendaRecord) {
      this.validateAddenda(dto.addendaRecord);
    }

    // Business rule validations
    this.validateBusinessRules(dto);
  }

  /**
   * Validate business rules
   */
  private validateBusinessRules(dto: CreateAchDto): void {
    // Rule 1: WEB and TEL SEC codes require specific addenda
    if ((dto.secCode === 'WEB' || dto.secCode === 'TEL') && !dto.addendaRecord) {
      throw new BadRequestException(`${dto.secCode} transactions require addenda information`);
    }

    // Rule 2: Same-day ACH has amount limits
    if (dto.sameDayAch && dto.amountCents > 100000000) {
      throw new BadRequestException('Same-day ACH transactions cannot exceed $1,000,000');
    }

    // Rule 3: Debit transactions require additional validation
    if (dto.transactionType === 'DEBIT') {
      // Ensure proper authorization is documented
      if (!dto.addendaRecord || !dto.addendaRecord.includes('AUTH')) {
        throw new BadRequestException('Debit transactions require authorization documentation in addenda');
      }
    }

    // Rule 4: CTX requires addenda
    if (dto.secCode === 'CTX' && !dto.addendaRecord) {
      throw new BadRequestException('CTX transactions require addenda information');
    }
  }

  /**
   * Validate batch of ACH payments
   */
  async validateAchBatch(payments: CreateAchDto[]): Promise<void> {
    if (!payments || payments.length === 0) {
      throw new BadRequestException('Batch must contain at least one payment');
    }

    if (payments.length > 10000) {
      throw new BadRequestException('Batch cannot exceed 10,000 payments');
    }

    // Validate each payment in the batch
    for (let i = 0; i < payments.length; i++) {
      try {
        await this.validateAchPayment(payments[i]);
      } catch (error) {
        throw new BadRequestException(`Payment ${i + 1} validation failed: ${error.message}`);
      }
    }

    // Validate batch-level rules
    this.validateBatchRules(payments);
  }

  /**
   * Validate batch-level business rules
   */
  private validateBatchRules(payments: CreateAchDto[]): void {
    // Rule 1: All payments in batch must have same SEC code
    const secCodes = new Set(payments.map(p => p.secCode));
    if (secCodes.size > 1) {
      throw new BadRequestException('All payments in batch must have the same SEC code');
    }

    // Rule 2: All payments must have same transaction type
    const transactionTypes = new Set(payments.map(p => p.transactionType));
    if (transactionTypes.size > 1) {
      throw new BadRequestException('All payments in batch must have the same transaction type');
    }

    // Rule 3: Calculate total batch amount
    const totalAmount = payments.reduce((sum, p) => sum + p.amountCents, 0);
    const maxBatchAmount = 10000000000; // $100M max per batch

    if (totalAmount > maxBatchAmount) {
      throw new BadRequestException(`Batch total amount cannot exceed $${(maxBatchAmount / 100).toLocaleString()}`);
    }

    // Rule 4: Check for duplicate payments (same receiver account and amount)
    const paymentKeys = new Set<string>();
    for (const payment of payments) {
      const key = `${payment.receiverAccountNumber}-${payment.receiverRoutingNumber}-${payment.amountCents}`;
      if (paymentKeys.has(key)) {
        throw new BadRequestException('Batch contains duplicate payments (same account and amount)');
      }
      paymentKeys.add(key);
    }
  }

  /**
   * Validate prenote (zero-dollar test transaction)
   */
  validatePrenote(dto: CreateAchDto): boolean {
    if (dto.amountCents !== 0) {
      throw new BadRequestException('Prenote transactions must have zero amount');
    }

    if (!dto.addendaRecord || !dto.addendaRecord.includes('PRENOTE')) {
      throw new BadRequestException('Prenote transactions must include PRENOTE in addenda');
    }

    return true;
  }

  /**
   * Check if routing number is valid for ACH
   */
  isRoutingNumberAchEligible(routingNumber: string): boolean {
    // First two digits indicate the Federal Reserve district (01-12)
    const district = parseInt(routingNumber.substring(0, 2), 10);
    
    if (district < 1 || district > 12) {
      throw new BadRequestException('Routing number is not ACH eligible (invalid Federal Reserve district)');
    }

    return true;
  }

  /**
   * Validate cutoff time for same-day ACH
   */
  validateSameDayAchCutoff(sameDayAch: boolean): boolean {
    if (!sameDayAch) {
      return true;
    }

    const now = new Date();
    const cutoffHour = 14; // 2 PM ET cutoff for same-day ACH
    const cutoffMinute = 45;

    // Convert to ET (simplified - in production, use proper timezone library)
    const currentHour = now.getHours();
    const currentMinute = now.getMinutes();

    if (currentHour > cutoffHour || (currentHour === cutoffHour && currentMinute > cutoffMinute)) {
      throw new BadRequestException('Same-day ACH cutoff time (2:45 PM ET) has passed for today');
    }

    return true;
  }
}
